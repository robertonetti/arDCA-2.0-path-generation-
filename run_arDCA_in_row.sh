#!/bin/bash
# Script per eseguire arDCA su diversi dataset
# Per ogni run:
#   - Costruisce i percorsi di train e test in base al parametro d
#   - Crea una cartella di output "model_normal_graph_d=<d>"
#   - Lancia il comando e salva l'output (stdout e stderr) in "run.log" nella cartella di output

# Array dei dataset: i nomi dei file contengono due pattern:
# "data_evo_cond_d=" e "data_evo_cond_longer_d="
datasets=(
    "data_evo_cond_d=10"
    "data_evo_cond_d=20"
    "data_evo_cond_d=30"
    "data_evo_cond_d=50"
    "data_evo_cond_d=70"
    "data_evo_cond_d=90"
)

for ds in "${datasets[@]}"; do
    # Estrae il valore d (parte dopo "=")
    d_value=${ds#*=}
    
    # Costruzione dei percorsi per i file di train e test
    train_file="data/evo_condition_start/condition_order/saved_as_fasta_10k_seqs/${ds}_train.fasta"
    test_file="data/evo_condition_start/condition_order/saved_as_fasta_10k_seqs/${ds}_test.fasta"
    
    # Definisce la cartella di output
    output_folder="models_prediction_second/models_10k_seqs/model_d=${d_value}"
    
    # Crea la cartella di output se non esiste
    mkdir -p "$output_folder"
    
    # Mostra su terminale il dataset in esecuzione
    echo "Avvio run per d=${d_value}"
    echo "Train file: ${train_file}"
    echo "Test file:  ${test_file}"
    echo "Output:     ${output_folder}"
    
    # Esegue il comando arDCA e salva il log (stdout e stderr) in run.log
    arDCA train -d "$train_file" \
                -o "$output_folder" \
                --batch_size 200000 \
                --nepochs 10000 \
                --data_test "$test_file" \
                --epsconv 1e-2 \
                --no_reweighting \
                --mode "second"  2>&1 | tee "${output_folder}/run.log" #--path_graph graphs/evolution_PF00014/graph_ii_nofields.pth
    
    # Controlla se il comando è andato a buon fine
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Errore durante la run per d=${d_value}. Uscita dallo script."
        exit 1
    fi
done

echo "Tutte le run sono state completate."
