import csv



def get_smiles(path):
    smile_list = []
    with open(path) as fp:
        reader = csv.DictReader(fp)
        columns = reader.fieldnames
        for row in reader:
            smile = row[columns[0]]
            smile_list.append(smile)
    return smile_list


def get_smiles_props(path):
    smile_list, prop_list = [], []
    with open(path) as fp:
        reader = csv.DictReader(fp)
        columns = reader.fieldnames
        for row in reader:
            smile = row[columns[0]]
            prop = row[columns[1]]
            smile_list.append(smile)
            prop_list.append(float(prop))
    return smile_list, prop_list