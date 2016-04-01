package ImageProcessing;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;

public class ConvertImageToVector {

    ArrayList<BufferedImage> bufferedImages;

    public ConvertImageToVector(ArrayList<BufferedImage> bImages) {
        this.bufferedImages = bImages;
    }

    public ArrayList<ArrayList<Double>> convert() {
        ArrayList<ArrayList<Double>> result = new ArrayList<>();
        int r = 0;
        while (r < bufferedImages.size()) {
            ArrayList<Double> value = new ArrayList<>();
            value.add(1.0);
            int squarePixel1 = bufferedImages.get(r).getWidth() / 16;
            int squarePixel = bufferedImages.get(r).getHeight() / 16;
            for (int i = 0; i < 16; ++i) {
                double[] cnt = new double[16];
                for (int j = i * squarePixel; j < (i + 1) * squarePixel; ++j) {
                    for (int k = 0; k < 16; ++k) {
                        for (int t = k * squarePixel1; t < (k + 1) * squarePixel1; ++t) {
                            if (bufferedImages.get(r).getRGB(t, j) == Color.BLACK.getRGB()) {
                                ++cnt[k];
                            }
                        }
                    }
                }
                for (int j = 0; j < 16; ++j) {
                    if (cnt[j] >= ((double) squarePixel * squarePixel) / 16) {
                        value.add(1.0);
                    } else {
                        value.add(0.0);
                    }
                }
            }
            result.add(value);
            ++r;
        }
        return result;
    }
}
