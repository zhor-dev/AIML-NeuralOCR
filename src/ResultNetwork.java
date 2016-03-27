import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class ResultNetwork {

    private double [][][]weights;
    private Network network;

    public ResultNetwork(int inputSize, int []layerSize) {
        weights = new double[layerSize.length][][];
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
        network = new Network(inputSize, layerSize, function);
    }

    public int recognize(double []i) {
        double []out;
        for (int k = 0; k < network.getLayers().length; ++k) {
            weights[k] = new double[network.getLayers()[k].getNeurons().length]
                    [network.getLayers()[k].getNeurons()[0].getWeights().length];
        }
        setWeights();
        network.setInputs(i);
        out = network.networkOutput();
        double err = getErr(out, TrainNetwork.dOutput[0]);
        int index = 0;
        for (int j = 1; j < TrainNetwork.dOutput.length; ++j) {
            if (getErr(TrainNetwork.dOutput[j], out) < err) {
                index = j;
                err = getErr(TrainNetwork.dOutput[j], out);
            }
        }
        return index;
    }

    private void setWeights() {
        weights = BackPropagationNetwork.loadWeights(weights);
        network.setWeights(weights);
    }

    private double getErr(double []o1, double []o2) {
        double err = 0;
        for (int j = 0; j < o2.length; ++j) {
            err += Math.pow(o1[j] - o2[j], 2);
        }
        return err;
    }

    public double[] getInput(int index) {
        double []input = new double[257];
        BufferedReader buffReader = null;
        try {
            FileReader fileReader = new FileReader("src/numbers.txt");
            buffReader = new BufferedReader(fileReader);
            String line = "";
            for (int t = 0 ; t < index; ++t) {
                line = buffReader.readLine();
            }
            line = line.substring(0, line.length() - 21);
            input[0] = 1;
            for (int j = 1; j < input.length; ++j) {
                input[j] = line.charAt((j - 1) * 7) == '0' ? 0 : 1;
            }
        } catch(IOException e){
            e.printStackTrace();
        } finally {
            try {
                assert buffReader != null;
                buffReader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return input;
    }
}
