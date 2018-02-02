import os
import pandas
import scipy.io

basePath = '../../data/02_count_emergence';
trainPath = 'train';
testPath = 'test';
xlFileName = 'count_gt.xlsx';
matFileName = 'count_gt.mat';

def save_mat_from_excel(xlFileName, matFileName) :
    try :
        data = pandas.read_excel(xlFileName);
    except :
        print 'Cannot open file %s' % xlFileName;
        exit()

    print 'Saving column = "{}" in .mat format'.format(data.columns[1].rstrip());
    scipy.io.savemat(matFileName, {'counts' : data[data.columns[1]].values } );

def main() :
    save_mat_from_excel(os.path.join(basePath, trainPath, xlFileName),
                        os.path.join(basePath, trainPath, matFileName) );

    save_mat_from_excel(os.path.join(basePath, testPath, xlFileName),
                        os.path.join(basePath, testPath, matFileName) );

if __name__ == "__main__" :
    main();
