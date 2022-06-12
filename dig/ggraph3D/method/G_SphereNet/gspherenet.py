import os
import torch
import numpy as np
from .model import SphGen



class G_SphereNet():
    def __init__(self):
        super(G_SphereNet, self).__init__()
        self.model = None
    

    def get_model(self, model_conf_dict, checkpoint_path=None):
        if model_conf_dict['use_gpu'] and not torch.cuda.is_available():
            model_conf_dict['use_gpu'] = False
        self.model = SphGen(**model_conf_dict)
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path))
    

    def load_pretrain_model(self, path):
        self.model.load_state_dict(torch.load(path))
    

    def train(self, loader, lr, wd, max_epochs, model_conf_dict, checkpoint_path, save_interval, save_dir):
        self.get_model(model_conf_dict, checkpoint_path)
        self.model.train()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=wd)
        ce_loss = torch.nn.BCELoss()
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        
        for epoch in range(1, max_epochs+1):
            total_loss = 0
            for batch, data_batch in enumerate(loader):
                optimizer.zero_grad()
                
                if model_conf_dict['use_gpu']:
                    for key in data_batch:
                        data_batch[key] = data_batch[key].to('cuda')
                
                node_out, focus_score, dist_out, angle_out, torsion_out = self.model(data_batch)

                ll_node = torch.mean(1/2 * (node_out[0] ** 2) - node_out[1])
                ll_dist = torch.mean(1/2 * (dist_out[0] ** 2) - dist_out[1])
                ll_angle = torch.mean(1/2 * (angle_out[0] ** 2) - angle_out[1])
                ll_torsion = torch.mean(1/2 * (torsion_out[0] ** 2) - torsion_out[1])
                cannot_focus = data_batch['cannot_focus']
                focus_ce = ce_loss(focus_score, cannot_focus)

                loss = ll_node + ll_dist + ll_angle + ll_torsion + focus_ce
                loss.backward()
                optimizer.step()

                total_loss += loss.to('cpu').item()
                print('Training iteration {} | loss {}'.format(batch, loss.to('cpu').item()))

            avg_loss = total_loss / (batch + 1)
            print("Training | Average loss {}".format(avg_loss))
            
            if epoch % save_interval == 0:
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'model_ckpt_{}.pth'.format(epoch)))


    def generate(self, model_conf_dict, checkpoint_path, n_mols=1000, chunk_size=100, num_min_node=7, num_max_node=25, temperature=[1.0, 1.0, 1.0, 1.0], focus_th=0.5):
        self.get_model(model_conf_dict, checkpoint_path)
        self.model.eval()

        type_to_atomic_number = np.array([1, 6, 7, 8, 9])
        mol_dicts = {}
        num_remain, one_time_gen = n_mols, chunk_size

        while num_remain > 0:
            if num_remain > one_time_gen:
                mols = self.model.generate(type_to_atomic_number, one_time_gen, temperature, num_min_node, num_max_node, focus_th)
            else:
                mols = self.model.generate(type_to_atomic_number, num_remain, temperature, num_min_node, num_max_node, focus_th)
            
            for num_atom in mols:
                if not num_atom in mol_dicts.keys():
                    mol_dicts[num_atom] = mols[num_atom]
                else:
                    mol_dicts[num_atom]['_atomic_numbers'] = np.concatenate((mol_dicts[num_atom]['_atomic_numbers'], mols[num_atom]['_atomic_numbers']), axis=0)
                    mol_dicts[num_atom]['_positions'] = np.concatenate((mol_dicts[num_atom]['_positions'], mols[num_atom]['_positions']), axis=0)
                    mol_dicts[num_atom]['_focus'] = np.concatenate((mol_dicts[num_atom]['_focus'], mols[num_atom]['_focus']), axis=0)
                num_mol = len(mols[num_atom]['_atomic_numbers'])
                num_remain -= num_mol
            
            print('{} molecules are generated!'.format(n_mols - num_remain))
        
        return mol_dicts