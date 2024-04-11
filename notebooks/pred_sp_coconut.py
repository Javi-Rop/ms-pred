import pandas as pd
import numpy as np
from collections import defaultdict
import ms_pred.magma.fragmentation as fe
from ms_pred.common.plot_utils import *
from ms_pred.dag_pred import joint_model


test_smiles = "O=C(O)C=1C(=O)C(O)(CC(=O)C1N)C2OC(COC(=O)C)C(OC(=O)C(N=CS)=CC)C(OC3OC(C)C(O)C(OC)C3)C2O"
test_ionization = "[M+H]+"
inten_ckpt = f"../quickstart/iceberg/models/nist_iceberg_score.ckpt"
gen_ckpt = f"../quickstart/iceberg/models/nist_iceberg_generate.ckpt"

# Load joint model
model = joint_model.JointModel.from_checkpoints(
    inten_checkpoint=inten_ckpt, gen_checkpoint=gen_ckpt
)
outputs = model.predict_mol(
    smi=test_smiles,
    adduct=test_ionization,
    device="cpu",
    max_nodes=100,
    binned_out=False,
    threshold=0,
)
root_inchi = outputs["root_inchi"]
frags = outputs["frags"]

# Generate a fragmentation engine
engine = fe.FragmentEngine(mol_str=root_inchi, mol_str_type="inchi")
# Convert from frags dict into a list of mz, inten
mass_to_obj = defaultdict(lambda: {})
for k, val in frags.items():
    masses, intens, form  = val["mz_charge"], val["intens"], val["form"]
    for m, i in zip(masses, intens):
        if i <= 0:
            continue
        cur_obj = mass_to_obj[m]
        if cur_obj.get("inten", 0) > 0:
            # update
            if cur_obj.get("inten") < i:
                cur_obj["frag_hash"] = k
                cur_obj['form'] = form
            cur_obj["inten"] += i
        else:
            cur_obj["inten"] = i
            cur_obj["frag_hash"] = k
            cur_obj['form'] = form

max_inten = max(*[i["inten"] for i in mass_to_obj.values()], 1e-9)
mass_to_obj = {
    k: dict(inten=v["inten"] / max_inten, frag_hash=v["frag_hash"], 
            form=v['form'])
            
    for k, v in mass_to_obj.items()
}
# Ordenar los fragmentos por intensidad en orden descendente y seleccionar los 10 primeros
top_10_fragments = sorted(mass_to_obj.items(), key=lambda x: x[1]["inten"], reverse=True)[:10]

# Imprimir los 10 fragmentos más intensos
for mz, fragment_info in top_10_fragments:
    inten = fragment_info['inten']
    frag_hash = fragment_info['frag_hash']
    form = fragment_info['form']
    print(f"MZ: {mz}, Intensidad: {inten}, Fragmento Hash: {frag_hash}, Fórmula: {form}")
    # Obtener los MZ e intensidades de los fragmentos en listas separadas
fragment_mz = [item[0] for item in sorted(list(mass_to_obj.items()), key=lambda x: x[1]["inten"], reverse=True)[:10]]
fragment_intensidades = [item[1]['inten'] for item in sorted(list(mass_to_obj.items()), key=lambda x: x[1]["inten"], reverse=True)[:10]]



ruta_csv = "/data/home/javier_rodriguez/ms2net/quimicaR/pruebas_Rmd/coconut_df.csv"


df = pd.read_csv(ruta_csv)


column_names = df.columns
print(column_names)
inten_ckpt = f"../quickstart/iceberg/models/nist_iceberg_score.ckpt"
gen_ckpt = f"../quickstart/iceberg/models/nist_iceberg_generate.ckpt"
# inten_ckpt = f"../quickstart/iceberg/models/canopus_iceberg_score.ckpt"
# gen_ckpt = f"../quickstart/iceberg/models/canopus_iceberg_generate.ckpt"

# Load joint model
model = joint_model.JointModel.from_checkpoints(
    inten_checkpoint=inten_ckpt, gen_checkpoint=gen_ckpt
)

df = df.drop(df.index[0])
df = df[df['molecular_weight'] <= 800]
df.columns = ['smiles' if col == 'SMILES' else col for col in df.columns]

from collections import defaultdict

def iceberg(test_smiles, test_ionization):
    try:
        inten_ckpt = f"/home/javier_rodriguez/ms2net/mspred/ms-pred/quickstart/iceberg/models/nist_iceberg_score.ckpt"
        gen_ckpt = f"/home/javier_rodriguez/ms2net/mspred/ms-pred/quickstart/iceberg/models/nist_iceberg_generate.ckpt"
        # inten_ckpt = f"../quickstart/iceberg/models/canopus_iceberg_score.ckpt"
        # gen_ckpt = f"../quickstart/iceberg/models/canopus_iceberg_generate.ckpt"
        model = joint_model.JointModel.from_checkpoints(
            inten_checkpoint=inten_ckpt, gen_checkpoint=gen_ckpt)
        outputs = model.predict_mol(
            smi=test_smiles,
            adduct=test_ionization,
            device="cpu",
            max_nodes=100,
            binned_out=False,
            threshold=0,
        )
        root_inchi = outputs["root_inchi"]
        frags = outputs["frags"]
        # Generate a fragmentation engine
        engine = fe.FragmentEngine(mol_str=root_inchi, mol_str_type="inchi")

        # Convert from frags dict into a list of mz, inten
        mass_to_obj = defaultdict(lambda: {})
        for k, val in frags.items():
            masses, intens, form  = val["mz_charge"], val["intens"], val["form"]
            for m, i in zip(masses, intens):
                if i <= 0:
                    continue
                cur_obj = mass_to_obj[m]
                if cur_obj.get("inten", 0) > 0:
                    # update
                    if cur_obj.get("inten") < i:
                        cur_obj["frag_hash"] = k
                        cur_obj['form'] = form
                    cur_obj["inten"] += i
                else:
                    cur_obj["inten"] = i
                    cur_obj["frag_hash"] = k
                    cur_obj['form'] = form

        max_inten = max([i["inten"] for i in mass_to_obj.values()] + [1e-9])
        mass_to_obj = {
            k: dict(inten=v["inten"] / max_inten, frag_hash=v["frag_hash"], 
                    form=v['form'])
                    
            for k, v in mass_to_obj.items()
        }

        # Ordenar los fragmentos por intensidad en orden descendente y seleccionar los 10 primeros
        top_10_fragments = sorted(mass_to_obj.items(), key=lambda x: x[1]["inten"], reverse=True)[:10]

        # Obtener los MZ e intensidades de los fragmentos en listas separadas
        fragment_mz = [item[0] for item in sorted(list(mass_to_obj.items()), key=lambda x: x[1]["inten"], reverse=True)[:]]
        fragment_intensidades = [item[1]['inten'] for item in sorted(list(mass_to_obj.items()), key=lambda x: x[1]["inten"], reverse=True)[:]]
        return fragment_mz, fragment_intensidades
    except Exception as e:
        # Manejo del error, por ejemplo, imprimir un mensaje y devolver 'NA'
        print(f"Error: {e}")
        return 'NA', 'NA'
    



resultados = []

for index, row in df.iterrows():
    test_smiles = row['smiles']  # Asegúrate de tener una columna "SMILES" en tu DataFrame
    # print(test_smiles)
    # test_ionization = row['Precursor_type']
    test_ionization = "[M+H]+"

    # Inicializa variables para los resultados
    fragment_mz = None
    fragment_intensidades = None

    # Verificar si test_smiles es una cadena antes de llamar a iceberg
    if isinstance(test_smiles, str):
        try:
            fragment_mz, fragment_intensidades = iceberg(test_smiles, test_ionization)
        except KeyError as e:
            # Manejar la excepción (puedes imprimir un mensaje, omitir la fila, etc.)
            print(f"Error: {e}")

    # Agregar los resultados al DataFrame resultados_df
    resultados.append({
        'SMILES': test_smiles,
        'FragmentMZ': fragment_mz,
        'FragmentIntensidades': fragment_intensidades
    })

# Convierte la lista de resultados en un nuevo DataFrame
resultados_df = pd.DataFrame(resultados)

# Añadir la columna de resultados al DataFrame original 'df'
df['FragmentMZ'] = resultados_df['FragmentMZ']
df['FragmentIntensidades'] = resultados_df['FragmentIntensidades']



df_prueba.to_csv('coconut_MH.csv', index=False)