from fmpairing_bert import Fmp_Bert
from matplotlib import pyplot as plt
from datetime import datetime
import argparse, os, csv, statistics

if __name__ == '__main__':
    # input the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('id', type=str, help='the name of the directory the result will be stored. If there are no directory of this name, a new one will be created.')
    parser.add_argument('data_directory', type=str, help='the directory that stores the dataset.')
    parser.add_argument('--model_name', type=str, default='bert-base-multilingual-cased', help='the name of BERT model you want to use. DEFAULT="bert-base-multilingual-cased"')
    parser.add_argument('--minimum_frequency', type=int, default=10, help='the number of the minimum frequency at which a subword in question needs to occur. DEFAULT=10')
    parser.add_argument('--brake_trials', type=int, default=10, help='the number of trials for which you want to stop the processes if the number pf clusters does not vary. DEFAULT=10')
    parser.add_argument('--output', type=bool, default=False, help='If True, the probabilities and entropies of each subword of each language will be output.')
    args = parser.parse_args()

    # contain the arguments into the variables 
    id = args.id
    dir = args.data_directory
    model = args.model_name
    min = args.minimum_frequency
    brake = args.brake_trials
    output = args.output

    # dictionary={language : average_entropy}
    avr_entropies = {}

    # create the directory that will store the result
    if 'results' not in os.listdir(os.getcwd):
        os.mkdir('results')

    # load the files 
    files = os.listdir(dir)
    for file in files:
        start = datetime.now()
        # estimate the number of meanings per subword and calculate the average entropy by using the class(fmpairing_bert.py)
        fmp_bert = Fmp_Bert(model, f'{dir}/{file}')
        fmp_bert.embed()
        fmp_bert.cluster(min, brake)
        avr_entropies[file[:-4]] = statistics.mean(fmp_bert.entropy.values())

        # record the output (optional)
        if output == True:
            if id not in os.listdir('results'):
                os.mkdir(f'results/{id}')
            with open(f'results/{id}/prob_{file[:-4]}.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                for key, values in fmp_bert.prob.items():
                    writer.writerow([key] + values)
            with open(f'results/{id}/entropy_{file[:-4]}.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['subword', 'entropy'])
                for key, values in fmp_bert.entropy.items():
                    writer.writerow([key, values])
        
        # output the progress
        end = datetime.now()
        time = end - start
        print(f'{file} is done. ({time.seconds} seconds)')
    
    # plot the result with a scatter plot and store as an image
    sorted_keys = sorted(avr_entropies.keys(), reverse=True)
    sorted_values = [avr_entropies[key] for key in sorted_keys]
    y_pos = range(len(sorted_keys))
    plt.figure(figsize=(10, 10))
    plt.scatter(sorted_values, y_pos)
    plt.yticks(y_pos, sorted_keys)
    plt.xlabel('Entropy')
    plt.ylabel('Language')
    plt.title(f'{id}')
    plt.savefig(f'results/{id}_entropies.png')
    plt.show()