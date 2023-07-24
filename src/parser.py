import argparse

parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--dataset', 
					metavar='-d', 
					type=str, 
					required=False,
					default='synthetic',
                    help="dataset from ['synthetic', 'SMD']")
parser.add_argument('--model', 
					metavar='-m', 
					type=str, 
					required=False,
					default='LSTM_Multivariate',
                    help="model name")
parser.add_argument('--test', 
					action='store_true', 
					help="test the model")
parser.add_argument('--retrain', 
					action='store_true', 
					help="retrain the model")
parser.add_argument('--less', 
					action='store_true', 
					help="train using less data")
parser.add_argument('--epochs',
					metavar='-e',
					type=int,
					required=True,
					default=50,
					help='Num epochs for training')
parser.add_argument('--modo',
					metavar='-mo',
					type=str,
					required=True,
					default=8,
					help='Execute or train with CIRCESiamese or TranAD')
parser.add_argument('--prod',
					metavar='-prod',
					type=str,
					required=True,
					default='False',
					help='Experimento E')
args = parser.parse_args()