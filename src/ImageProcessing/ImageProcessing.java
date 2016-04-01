package ImageProcessing;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class ImageProcessing {

    private byte []imageArray;
    private int threshold;
    private BufferedImage image;

    public ImageProcessing(byte []iArray, int t) {
        this.imageArray = iArray;
        this.threshold = t;
    }

    public ImageProcessing(BufferedImage i, int t) {
        this.image = i;
        this.threshold = t;
    }

    public ArrayList<ArrayList<Double>> getCharacters() throws IOException {
        return (new ConvertImageToVector(getResultImages())).convert();
    }

    private ArrayList<BufferedImage> getResultImages() throws IOException {
        return (new RecognizeImageParts((new CropImage((new GetBlackAndWhiteImage(image))
                .changeToBlackWhite(threshold))).cropBWImage())).getImageDetectedParts();
    }

    public static void main(String... args) {
        BufferedImage image = null;
        try {
            image = ImageIO.read(new File("C:\\five.jpg"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        ImageProcessing imageProcessing = new ImageProcessing(image, 100);
        try {
            ArrayList<ArrayList<Double>> res = imageProcessing.getCharacters();
            String resString = "";
            for (ArrayList<Double> aVal : res) {
                for (int j = 1; j < 17; ++j) {
                    for (int k = (j - 1) * 16; k < j * 16; ++k) {
                        if (aVal.get(k) == 1.0) {
                            resString += 1;
                        } else {
                            resString += 0;
                        }
                    }
                    resString += "\n";
                }
            }
            System.out.println(resString);
            /*ImageIO.write((new RecognizeImageParts((new CropImage((new GetBlackAndWhiteImage(image))
                    .changeToBlackWhite(100))).cropBWImage())).recognizeBWImageParts().get(3),
                    "jpg", new File("C:/Users/TOSHIBA/Documents/Sample.jpg"));*/
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
