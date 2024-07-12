import csv
import os



  # Indices des colonnes à extraire des fichiers de wholebody pour avoir les mains
colonnes_a_extraire = [182, 183, 184, 185, 186, 187, 192, 193, 200, 201, 208, 209, 216, 217, 224, 225, 226, 227, 228, 229, 234, 235, 242, 243,
    250, 251, 258, 259
    ]

# Fonction pour lire et extraire des colonnes spécifiques d'un fichier
def lire_et_extraire_colonnes(fichier, indices):
    donnees = []
    with open(fichier, 'r') as f:
        lecteur = csv.reader(f, delimiter=',')
        for ligne in lecteur:
            colonne_extraite = [ligne[i] for i in indices if i < len(ligne)]
            donnees.append(colonne_extraite)
    return donnees

# Fonction pour lire tout un fichier
def lire_fichier(fichier):
    donnees = []
    with open(fichier, 'r') as f:
        lecteur = csv.reader(f, delimiter=',')
        for ligne in lecteur:
            donnees.append(ligne)
    return donnees


# Itérer sur les fichiers des deux listes en parallèle
for fichier_base, fichier_main in zip(liste_fichiers, liste_fichiers_main):
    # Lire le fichier de base
    donnees_base = lire_fichier(fichier_base)
    
    # Lire et extraire les colonnes du fichier main
    donnees_extraite = lire_et_extraire_colonnes(fichier_main, colonnes_a_extraire)
    
    # Vérifier que les fichiers ont le même nombre de lignes
    if len(donnees_base) != len(donnees_extraite):
        print(f"Erreur : Les fichiers {fichier_base} et {fichier_main} n'ont pas le même nombre de lignes.")
        continue

    # Ajouter les colonnes extraites à chaque ligne du fichier de base
    donnees_combinees = [base + extraite for base, extraite in zip(donnees_base, donnees_extraite)]
    
    # Créer le chemin du fichier de sortie avec "all" à la place de "wholebody"
    fichier_sortie = fichier_main.replace("wholebody", "all")
    
    # S'assurer que le répertoire de sortie existe
    os.makedirs(os.path.dirname(fichier_sortie), exist_ok=True)
    
    # Sauvegarder le fichier combiné
    with open(fichier_sortie, 'w', newline='') as f:
        ecrivain = csv.writer(f, delimiter=',')
        ecrivain.writerows(donnees_combinees)


# d = np.loadtxt('/home/tbousquet/Documents/Challenge/Donnees_mmpose_avec_score/all/result_'+trial+'_26585_'+subject+'.txt', delimiter = ',')
# print(d[1].shape)

# liste_fichiers_all = [
#     '/home/tbousquet/Documents/Challenge/Donnees_mmpose_avec_score/all/result_'+trial+'_26585_'+subject+'.txt',
#     '/home/tbousquet/Documents/Challenge/Donnees_mmpose_avec_score/all/result_'+trial+'_26587_'+subject+'.txt',
#     '/home/tbousquet/Documents/Challenge/Donnees_mmpose_avec_score/all/result_'+trial+'_26578_'+subject+'.txt',
#     '/home/tbousquet/Documents/Challenge/Donnees_mmpose_avec_score/all/result_'+trial+'_26579_'+subject+'.txt',
#     '/home/tbousquet/Documents/Challenge/Donnees_mmpose_avec_score/all/result_'+trial+'_26580_'+subject+'.txt',
#     '/home/tbousquet/Documents/Challenge/Donnees_mmpose_avec_score/all/result_'+trial+'_26582_'+subject+'.txt',
#     '/home/tbousquet/Documents/Challenge/Donnees_mmpose_avec_score/all/result_'+trial+'_26583_'+subject+'.txt',
#     '/home/tbousquet/Documents/Challenge/Donnees_mmpose_avec_score/all/result_'+trial+'_26584_'+subject+'.txt',
#     '/home/tbousquet/Documents/Challenge/Donnees_mmpose_avec_score/all/result_'+trial+'_26586_'+subject+'.txt'
# ]
