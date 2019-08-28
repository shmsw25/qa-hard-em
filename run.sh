data=$1
loss=$2
bs=60
pbs=600

if [ ${loss} = "hard-em" ]
then
    tau=$3
    output_dir="out/${data}-${loss}-${tau}"
else
    output_dir="out/${data}-${loss}"
    tau=0
fi

train_file="open-domain-qa-data/${data}-train0.json"
for index in 1 2 3 ; do
    train_file="${train_file},open-domain-qa-data/${data}-train${index}.json"
done
dev_file="open-domain-qa-data/${data}-dev.json"
test_file="open-domain-qa-data/${data}-test.json"

python3 main.py --do_train --output_dir ${output_dir} \
          --train_file ${train_file} --predict_file ${dev_file} \
          --train_batch_size ${bs} --predict_batch_size ${pbs} --loss_type ${loss} --tau ${tau}
python3 main.py --do_predict --output_dir ${output_dir} \
          --predict_file ${dev_file} \
          --init_checkpoint ${output_dir}/best-model.pt \
          --predict_batch_size ${pbs} --n_paragraphs "10,20,40,80" --prefix dev_
python3 main.py --do_predict --output_dir ${output_dir} \
          --predict_file ${test_file} \
          --init_checkpoint ${output_dir}/best-model.pt \
          --predict_batch_size ${pbs} --n_paragraphs "10,20,40,80" --prefix test_


