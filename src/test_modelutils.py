from modelutils import modelutils
from clean import simple_clean
from clean import complex_clean
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Spring 2019')
    parser.add_argument('--decision', default="../text/history/RatesDecision.csv")
    parser.add_argument('--minutes', default="../text/minutes")
    parser.add_argument('--speeches', default="../text/speeches")
    parser.add_argument('--statements', default="../text/statements")
    parser.add_argument('--pctTrain', default=0.75, type=float)
    parser.add_argument('--Niter', default=10, type=int)
    parser.add_argument('--cleanAlgo', default="complex")
    parser.add_argument('--ngram', nargs='+', default=['1,1'])
    args = parser.parse_args()

    df = modelutils.decisionDF(args.decision)
    mn = modelutils.getMinutes(args.minutes, df, complex_clean)
    st = modelutils.getStatements(args.statements, df, complex_clean)
    sp = modelutils.getSpeeches(args.speeches, df, complex_clean)

    if False:
        print("--RatesDecision")
        print(df)
        print("---Minutes")
        print(mn)
        print("---Statements")
        print(st)
        print("---Speeches")
        print(sp)

    train, test = modelutils.splitTrainTest([mn, st, sp], 0.75)
    print("train_size=%d, test_size=%d, total_size=%d" % (len(train), len(test), len(train) + len(test)))
    print("randmoly split training set")
    print(train)
    print("randmoly split test set")
    print(test)

    #
    #
    #
    data_stacked = modelutils.stackFeatures([mn, st, sp])
    print("--- all of the features stacked---")
    print(data_stacked)

