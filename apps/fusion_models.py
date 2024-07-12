import os



no_sujet = int(input("Entrez le numéro du sujet (ex: 2): "))
task = input("Entrez la tâche (ex: 'assis-debout'): ")
data_path='/home/tbousquet/Documents/Donnees_cosmik/Data/'

#Etape 1 : Transformation des fichiers pour ajouter les coordonées des mains à partir des fichiers de body26 et wholebody.


#Ces chemins ont été déterminés lors du process de MMPose
liste_fichiers = [
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26585_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26587_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26578_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26579_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26580_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26582_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26583_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26584_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26586_sujet' + str(no_sujet) + '.txt'
    ]

liste_fichiers_main = [
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26585_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26587_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26578_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26579_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26580_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26582_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26583_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26584_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26586_sujet' + str(no_sujet) + '.txt'
]

if not os.path.exists  ( f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/all') :
    try:    
        with open('utils/add_hands.py') as f:
            code = f.read()
            exec(code)
            print ('Les coordonnées correspondantes aux mains ont été rajoutées.')

    except Exception as e:
        print(f"Erreur lors de l'exécution de 'add_hands.py' : {e}")

else : 
    print('Le dossier regroupant les coordonnées du corps et des mains a été récupéré.')



