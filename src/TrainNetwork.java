import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class TrainNetwork {

    private BackPropagationNetwork network;

    public static final double [][]dOutput = {
            {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
    };
    public TrainNetwork(int intputSize, int []lSize, double epsilon, double alpha) {
        ActivationFunction function = new ActivationFunction() {
            @Override
            public double activationFunction(double S) {
                return ActivationFunctions.sigmoid(S);
            }

            @Override
            public double functionDerivative(double S) {
                return ActivationFunctions.derivativeSigmoid(S);
            }
        };
        network = new BackPropagationNetwork(intputSize, lSize, dOutput, function);
        network.setEpsilon(epsilon);
        network.setAlpha(alpha);
    }

    public void train(int trainCycle) {
        for (int j = 0; j < trainCycle; ++j) {
            getNetworkDataAndTrain();
        }
        network.saveWeights();
    }

    private void getNetworkDataAndTrain() {
        BufferedReader buffreader = null;
        try {
            FileReader fileReader = new FileReader("src/numbers.txt");
            buffreader = new BufferedReader(fileReader);
            String line = buffreader.readLine();
            double []desiredOutput = new double[10];
            double []inputs = new double[257];
            while (line != null) {
                String str = line.substring(line.length() - 21);
                line = line.substring(0, line.length() - 21);
                for (int j = 0; j < desiredOutput.length; ++j) {
                    desiredOutput[j] = str.charAt(j * 2 + 1) == '0' ? 0 : 1;
                }
                inputs[0] = 1;
                for (int j = 1; j < inputs.length; ++j) {
                    inputs[j] = line.charAt((j - 1) * 7) == '0' ? 0 : 1;
                }
                network.setDesiredOutput(desiredOutput);
                network.setInputs(inputs);
                network.trainNetwork();
                line = buffreader.readLine();
            }
        } catch(IOException e){
            e.printStackTrace();
        } finally {
            try {
                assert buffreader != null;
                buffreader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

}
