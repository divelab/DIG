import torch
import torch.nn as nn
from .mol_tree import Vocab, MolTree
from .nnutils import create_var
from .jtnn_enc_bo import JTNNEncoderBO
from .jtnn_dec import JTNNDecoder
from .mpn import MPN, mol2graph
from .jtmpn import JTMPN

from .chemutils import enum_assemble, set_atommap, copy_edit_mol, attach_mols, atom_equal, decode_stereo
import rdkit
import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import copy
import math


def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1


class JTNNVAEBO(nn.Module):

    def __init__(self, vocab, hidden_size, latent_size, depth, stereo=True):
        super(JTNNVAEBO, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth = depth

        self.embedding = nn.Embedding(vocab.size(), hidden_size)
        self.jtnn = JTNNEncoderBO(vocab, hidden_size, self.embedding)
        self.jtmpn = JTMPN(hidden_size, depth)
        self.mpn = MPN(hidden_size, depth)
        self.decoder = JTNNDecoderBO(
            vocab, hidden_size, latent_size / 2, self.embedding)

        self.T_mean = nn.Linear(hidden_size, latent_size / 2)
        self.T_var = nn.Linear(hidden_size, latent_size / 2)
        self.G_mean = nn.Linear(hidden_size, latent_size / 2)
        self.G_var = nn.Linear(hidden_size, latent_size / 2)

        self.assm_loss = nn.CrossEntropyLoss(size_average=False)
        self.use_stereo = stereo
        if stereo:
            self.stereo_loss = nn.CrossEntropyLoss(size_average=False)

    def encode(self, mol_batch):
        set_batch_nodeID(mol_batch, self.vocab)
        root_batch = [mol_tree.nodes[0] for mol_tree in mol_batch]
        tree_mess, tree_vec = self.jtnn(root_batch)

        smiles_batch = [mol_tree.smiles for mol_tree in mol_batch]
        mol_vec = self.mpn(mol2graph(smiles_batch))
        return tree_mess, tree_vec, mol_vec

    def encode_latent_mean(self, smiles_list):
        mol_batch = [MolTree(s) for s in smiles_list]
        for mol_tree in mol_batch:
            mol_tree.recover()

        _, tree_vec, mol_vec = self.encode(mol_batch)
        tree_mean = self.T_mean(tree_vec)
        mol_mean = self.G_mean(mol_vec)
        return torch.cat([tree_mean, mol_mean], dim=1)

    def forward(self, mol_batch, beta=0):
        batch_size = len(mol_batch)

        tree_mess, tree_vec, mol_vec = self.encode(mol_batch)

        tree_mean = self.T_mean(tree_vec)
        # Following Mueller et al.
        tree_log_var = -torch.abs(self.T_var(tree_vec))
        mol_mean = self.G_mean(mol_vec)
        # Following Mueller et al.
        mol_log_var = -torch.abs(self.G_var(mol_vec))

        z_mean = torch.cat([tree_mean, mol_mean], dim=1)
        z_log_var = torch.cat([tree_log_var, mol_log_var], dim=1)
        kl_loss = -0.5 * \
            torch.sum(1.0 + z_log_var - z_mean * z_mean -
                      torch.exp(z_log_var)) / batch_size

        epsilon = create_var(torch.randn(
            batch_size, self.latent_size / 2), False)
        tree_vec = tree_mean + torch.exp(tree_log_var / 2) * epsilon
        epsilon = create_var(torch.randn(
            batch_size, self.latent_size / 2), False)
        mol_vec = mol_mean + torch.exp(mol_log_var / 2) * epsilon

        word_loss, topo_loss, word_acc, topo_acc = self.decoder(
            mol_batch, tree_vec)
        assm_loss, assm_acc = self.assm(mol_batch, mol_vec, tree_mess)
        if self.use_stereo:
            stereo_loss, stereo_acc = self.stereo(mol_batch, mol_vec)
        else:
            stereo_loss, stereo_acc = 0, 0

        all_vec = torch.cat([tree_vec, mol_vec], dim=1)
        loss = word_loss + topo_loss + assm_loss + 2 * stereo_loss + beta * kl_loss

        return loss, kl_loss.item(), word_acc, topo_acc, assm_acc, stereo_acc

    def assm(self, mol_batch, mol_vec, tree_mess):
        cands = []
        batch_idx = []
        for i, mol_tree in enumerate(mol_batch):
            for node in mol_tree.nodes:
                # Leaf node's attachment is determined by neighboring node's attachment
                if node.is_leaf or len(node.cands) == 1:
                    continue
                cands.extend([(cand, mol_tree.nodes, node)
                             for cand in node.cand_mols])
                batch_idx.extend([i] * len(node.cands))

        cand_vec = self.jtmpn(cands, tree_mess)
        cand_vec = self.G_mean(cand_vec)

        batch_idx = create_var(torch.LongTensor(batch_idx))
        mol_vec = mol_vec.index_select(0, batch_idx)

        mol_vec = mol_vec.view(-1, 1, self.latent_size / 2)
        cand_vec = cand_vec.view(-1, self.latent_size / 2, 1)
        scores = torch.bmm(mol_vec, cand_vec).squeeze()

        cnt, tot, acc = 0, 0, 0
        all_loss = []
        for i, mol_tree in enumerate(mol_batch):
            comp_nodes = [node for node in mol_tree.nodes if len(
                node.cands) > 1 and not node.is_leaf]
            cnt += len(comp_nodes)
            for node in comp_nodes:
                label = node.cands.index(node.label)
                ncand = len(node.cands)
                cur_score = scores.narrow(0, tot, ncand)
                tot += ncand

                if cur_score[label].item() >= cur_score.max().item():
                    acc += 1

                label = create_var(torch.LongTensor([label]))
                all_loss.append(self.assm_loss(cur_score.view(1, -1), label))

        #all_loss = torch.stack(all_loss).sum() / len(mol_batch)
        all_loss = sum(all_loss) / len(mol_batch)
        return all_loss, acc * 1.0 / cnt

    def stereo(self, mol_batch, mol_vec):
        stereo_cands, batch_idx = [], []
        labels = []
        for i, mol_tree in enumerate(mol_batch):
            cands = mol_tree.stereo_cands
            if len(cands) == 1:
                continue
            if mol_tree.smiles3D not in cands:
                cands.append(mol_tree.smiles3D)
            stereo_cands.extend(cands)
            batch_idx.extend([i] * len(cands))
            labels.append((cands.index(mol_tree.smiles3D), len(cands)))

        if len(labels) == 0:
            return create_var(torch.zeros(1)), 1.0

        batch_idx = create_var(torch.LongTensor(batch_idx))
        stereo_cands = self.mpn(mol2graph(stereo_cands))
        stereo_cands = self.G_mean(stereo_cands)
        stereo_labels = mol_vec.index_select(0, batch_idx)
        scores = torch.nn.CosineSimilarity()(stereo_cands, stereo_labels)

        st, acc = 0, 0
        all_loss = []
        for label, le in labels:
            cur_scores = scores.narrow(0, st, le)
            if cur_scores.data[label] >= cur_scores.max().data:
                acc += 1
            label = create_var(torch.LongTensor([label]))
            all_loss.append(self.stereo_loss(cur_scores.view(1, -1), label))
            st += le
        #all_loss = torch.cat(all_loss).sum() / len(labels)
        all_loss = sum(all_loss) / len(labels)
        return all_loss, acc * 1.0 / len(labels)

    def reconstruct(self, smiles, prob_decode=False):
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        _, tree_vec, mol_vec = self.encode([mol_tree])

        tree_mean = self.T_mean(tree_vec)
        # Following Mueller et al.
        tree_log_var = -torch.abs(self.T_var(tree_vec))
        mol_mean = self.G_mean(mol_vec)
        # Following Mueller et al.
        mol_log_var = -torch.abs(self.G_var(mol_vec))

        epsilon = create_var(torch.randn(1, self.latent_size / 2), False)
        tree_vec = tree_mean + torch.exp(tree_log_var / 2) * epsilon
        epsilon = create_var(torch.randn(1, self.latent_size / 2), False)
        mol_vec = mol_mean + torch.exp(mol_log_var / 2) * epsilon
        return self.decode(tree_vec, mol_vec, prob_decode)

    def recon_eval(self, smiles):
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        _, tree_vec, mol_vec = self.encode([mol_tree])

        tree_mean = self.T_mean(tree_vec)
        # Following Mueller et al.
        tree_log_var = -torch.abs(self.T_var(tree_vec))
        mol_mean = self.G_mean(mol_vec)
        # Following Mueller et al.
        mol_log_var = -torch.abs(self.G_var(mol_vec))

        all_smiles = []
        for i in range(10):
            epsilon = create_var(torch.randn(1, self.latent_size / 2), False)
            tree_vec = tree_mean + torch.exp(tree_log_var / 2) * epsilon
            epsilon = create_var(torch.randn(1, self.latent_size / 2), False)
            mol_vec = mol_mean + torch.exp(mol_log_var / 2) * epsilon
            for j in range(10):
                new_smiles = self.decode(tree_vec, mol_vec, prob_decode=True)
                all_smiles.append(new_smiles)
        return all_smiles

    def sample_prior(self, prob_decode=False):
        tree_vec = create_var(torch.randn(1, self.latent_size / 2), False)
        mol_vec = create_var(torch.randn(1, self.latent_size / 2), False)
        return self.decode(tree_vec, mol_vec, prob_decode)

    def sample_eval(self):
        tree_vec = create_var(torch.randn(1, self.latent_size / 2), False)
        mol_vec = create_var(torch.randn(1, self.latent_size / 2), False)
        all_smiles = []
        for i in range(100):
            s = self.decode(tree_vec, mol_vec, prob_decode=True)
            all_smiles.append(s)
        return all_smiles

    def decode(self, tree_vec, mol_vec, prob_decode):
        pred_root, pred_nodes = self.decoder.decode(tree_vec, prob_decode)

        # Mark nid & is_leaf & atommap
        for i, node in enumerate(pred_nodes):
            node.nid = i + 1
            node.is_leaf = (len(node.neighbors) == 1)
            if len(node.neighbors) > 1:
                set_atommap(node.mol, node.nid)

        tree_mess = self.jtnn([pred_root])[0]

        cur_mol = copy_edit_mol(pred_root.mol)
        global_amap = [{}] + [{} for node in pred_nodes]
        global_amap[1] = {atom.GetIdx(): atom.GetIdx()
                          for atom in cur_mol.GetAtoms()}

        cur_mol = self.dfs_assemble(tree_mess, mol_vec, pred_nodes, cur_mol, global_amap, [
        ], pred_root, None, prob_decode)
        if cur_mol is None:
            return None

        cur_mol = cur_mol.GetMol()
        set_atommap(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        if cur_mol is None:
            return None
        if self.use_stereo == False:
            return Chem.MolToSmiles(cur_mol)

        smiles2D = Chem.MolToSmiles(cur_mol)
        stereo_cands = decode_stereo(smiles2D)
        if len(stereo_cands) == 1:
            return stereo_cands[0]
        stereo_vecs = self.mpn(mol2graph(stereo_cands))
        stereo_vecs = self.G_mean(stereo_vecs)
        scores = nn.CosineSimilarity()(stereo_vecs, mol_vec)
        _, max_id = scores.max(dim=0)
        return stereo_cands[max_id.data[0]]

    def dfs_assemble(self, tree_mess, mol_vec, all_nodes, cur_mol, global_amap, fa_amap, cur_node, fa_node, prob_decode):
        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(
            neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cur_amap = [(fa_nid, a2, a1)
                    for nid, a1, a2 in fa_amap if nid == cur_node.nid]
        cands = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)
        if len(cands) == 0:
            return None
        cand_smiles, cand_mols, cand_amap = zip(*cands)

        cands = [(candmol, all_nodes, cur_node) for candmol in cand_mols]

        cand_vecs = self.jtmpn(cands, tree_mess)
        cand_vecs = self.G_mean(cand_vecs)
        mol_vec = mol_vec.squeeze()
        scores = torch.mv(cand_vecs, mol_vec) * 20

        if prob_decode:
            probs = nn.Softmax()(scores.view(1, -1)).squeeze() + 1e-5  # prevent prob = 0
            cand_idx = torch.multinomial(probs, probs.numel())
        else:
            _, cand_idx = torch.sort(scores, descending=True)

        backup_mol = Chem.RWMol(cur_mol)
        for i in range(cand_idx.numel()):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[cand_idx[i].item()]
            new_global_amap = copy.deepcopy(global_amap)

            for nei_id, ctr_atom, nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][ctr_atom]

            # father is already attached
            cur_mol = attach_mols(cur_mol, children, [], new_global_amap)
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None:
                continue

            result = True
            for nei_node in children:
                if nei_node.is_leaf:
                    continue
                cur_mol = self.dfs_assemble(
                    tree_mess, mol_vec, all_nodes, cur_mol, new_global_amap, pred_amap, nei_node, cur_node, prob_decode)
                if cur_mol is None:
                    result = False
                    break
            if result:
                return cur_mol

        return None
