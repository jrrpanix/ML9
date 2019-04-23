from modelutils import modelutils
from clean import simple_clean
from clean import complex_clean
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SocialNetworks CSV to HDF5')
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
    
    print("--RatesDecision")
    print(df)
    print("---Minutes")
    print(mn)
    print("---Statements")
    print(st)
    print("---Speeches")
    print(sp)

