import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;

public class SquareImages {

    private ArrayList<BufferedImage> images;

    public SquareImages(ArrayList<BufferedImage> iArray) {
        this.images = iArray;
    }

    private ArrayList<BufferedImage> squareCharacters() {
        int k = 0;
        ArrayList<BufferedImage> result = new ArrayList<>();
        while (k < images.size()) {
            int max = Math.max(images.get(k).getHeight(), images.get(k).getWidth());
            int min = Math.min(images.get(k).getHeight(), images.get(k).getWidth());
            BufferedImage changeBuff = new BufferedImage(max, max, images.get(k).getType());
            for (int i = 0; i < changeBuff.getHeight(); ++i) {
                for (int j = 0; j < changeBuff.getWidth(); ++j) {
                    changeBuff.setRGB(i, j, Color.WHITE.getRGB());
                }
            }
            int origin = (max - min) / 2;
            if (max == images.get(k).getHeight()) {
                for (int i = origin; i < origin + images.get(k).getWidth(); ++i) {
                    for (int j = 0; j < max; ++j) {
                        if (images.get(k).getRGB(j, i - origin) == Color.BLACK.getRGB()) {
                            changeBuff.setRGB(j, i , Color.BLACK.getRGB());
                        }
                    }
                }
            } else {
                for (int i = 0; i < max; ++i) {
                    for (int j = origin; j < origin + images.get(k).getHeight(); ++j) {
                        if (images.get(k).getRGB(j - origin, i) == Color.BLACK.getRGB()) {
                            changeBuff.setRGB(j, i , Color.BLACK.getRGB());
                        }
                    }
                }
            }
            result.add(changeBuff);
            ++k;
        }
        return result;
    }

}
