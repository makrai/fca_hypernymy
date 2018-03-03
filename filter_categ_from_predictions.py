import argparse
import logging
import os

from evaluate_dag import ThreeHundredSparsians
from evaluate_dag import get_args as parse_dummy_args

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s: (%(lineno)s) %(levelname)s %(message)s"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Filter concepts or entities from predictions')
    parser.add_argument(
        '--predictions',
        default='/mnt/store/friend/proj/SemEval18-hypernym/results/test/predictions/1A.english_1A_UMBC_tokenized.txt_100_sg.vec.gz_True_200_0.3_unit_True_vocabulary_filtered.alph.reduced2_more_permissive.dot_True.1.0.True.output.txt')
    parser.add_argument('--file_struct', choices=['szeged', 'sztaki'], default='szeged')
    # az 1.0 baloldalán álló az az, hogy az attpárok be voltak-e kapcsolva
    # a jobboldali pedig az h filterezve volt-e a candidate hipernímák halmaza
    # ez után az érték fog majd a következő verzióba egy olyan egész is
    # bekerülni, ami pedig a tanítás során látott negatív példák száma
    return parser.parse_args()


class CategoryFilter():
    def __init__(self):
        self.args = parse_args()
        self.pred_dir, self.pred_basefn = os.path.split(self.args.predictions)
        subtask = self.pred_basefn.split('_')[0]
        assert ('dev' in self.args.predictions) != (
            'test' in self.args.predictions)
        phase = 'trial' if 'dev' in self.args.predictions else 'test'
        proj_dir = '/mnt/store/friend/proj/SemEval18-hypernym/'

        def input_fn(data_or_gold):
            input_dir = os.path.join(
                proj_dir, 'SemEval18-Task9/{}/{}'.format(phase, data_or_gold))
            input_basefn = '{}.{}.{}.txt'.format(subtask, phase, data_or_gold)
            return input_dir, input_basefn
        query_dir, query_basefn = input_fn('data')
        gold_dir, gold_basefn = input_fn('gold')
        for categ in ['concept', 'entity']:
            for all_dir in query_dir, gold_dir:
                categ_dir = os.path.join(all_dir, categ)
                if not os.path.isdir(categ_dir):
                    os.mkdir(categ_dir)
            categ_dir = os.path.join(self.pred_dir, categ)
            if not os.path.isdir(categ_dir):
                    os.mkdir(categ_dir)
        self.query_path = os.path.join(query_dir, query_basefn)
        self.gold_path = os.path.join(gold_dir, gold_basefn)
        self.gold_concept_path = os.path.join(gold_dir, 'concept', gold_basefn)
        self.gold_entity_path = os.path.join(gold_dir, 'entity', gold_basefn)
        self.pred_concept_path = os.path.join(self.pred_dir, 'concept',
                                              self.pred_basefn)
        self.pred_entity_path = os.path.join(self.pred_dir, 'entity',
                                             self.pred_basefn)

    def filter_by_categ(self):
        logging.info('Filtering...')
        with open(self.query_path) as query_f, \
                open(self.gold_path) as gold_f, \
                open(self.args.predictions) as pred_f, \
                open(self.gold_concept_path, mode='w') as gold_concept_f, \
                open(self.gold_entity_path, mode='w') as gold_entity_f, \
                open(self.pred_concept_path, mode='w') as pred_concept_f, \
                open(self.pred_entity_path, mode='w') as pred_entity_f:
            for line in query_f:
                categ = line.strip().split('\t')[1]
                gold_l = gold_f.readline()
                pred_l = pred_f.readline()
                if categ == 'Concept':
                    gold_concept_f.write(gold_l)
                    pred_concept_f.write(pred_l)
                else:
                    gold_entity_f.write(gold_l)
                    pred_entity_f.write(pred_l)

    def write_metrics_wrapper(self):
        if not os.path.isfile(self.gold_concept_path):  #TODO
            self.filter_by_categ()
        # TODO The wrapper around write_metrics is done in a hacky way,
        # both argparsers run. In order for this code to work, a dummy
        # '--predictions' argument should be added to the argparser in
        # evaluate_dag
        dummy_args = parse_dummy_args()
        dummy_args.file_struct = self.args.file_struct
        sparsians = ThreeHundredSparsians(dummy_args)
        sparsians.regularization = 1
        for categ in ['concept', 'entity']:
            metrics_categ = os.path.join(self.pred_dir,
                                         '../metrics/{}'.format(categ))
            if not os.path.isdir(metrics_categ):
                os.mkdir(metrics_categ)
        metric_filen_patt = os.path.join(self.pred_dir, '../metrics/{}',
                                         self.pred_basefn)
        logging.info('Writing metrics...')
        sparsians.write_metrics(self.gold_path, self.args.predictions,
                                metric_filen_patt.format(''))
        sparsians.write_metrics(self.gold_concept_path, self.pred_concept_path,
                                metric_filen_patt.format('concept'))
        sparsians.write_metrics(self.gold_entity_path, self.pred_entity_path,
                                metric_filen_patt.format('entity'))


if __name__ == '__main__':
    cfilter = CategoryFilter() 
    cfilter.write_metrics_wrapper()
