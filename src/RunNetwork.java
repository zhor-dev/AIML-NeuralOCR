import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class RunNetwork {
    public static void main(String... args) {
        int []lSize = {139, 10};
        //TrainNetwork b = new TrainNetwork(257, lSize, 1, 1);
        //b.train(30);
        //ResultNetwork network = new ResultNetwork(lSize);
        //for (int j = 1; j <= 160; ++j) {
        //    System.out.println(network.recognize(network.getInput(j)));
        //}
        BufferedImage image = null;
        try {
            image = ImageIO.read(new File("C:\\five.jpg"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        ImageProcessing imageProcessing = new ImageProcessing(image, 100);
        try {
            ArrayList<ArrayList<Double>> res = imageProcessing.getCharacters();
            double arr[] = new double[257];
            for (int j = 0; j < arr.length; ++j) {
                arr[j] = res.get(0).get(j);
                System.out.print((arr[j] == 1 ? 1 : 0));
                if (j % 16 == 0) {
                    System.out.println();
                }
            }
            ResultNetwork network = new ResultNetwork(257, lSize);
            System.out.println(network.recognize(arr));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
