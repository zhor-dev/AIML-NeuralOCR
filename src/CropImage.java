import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;

public class CropImage {

    private BufferedImage BWImage;

    public CropImage(byte []BWImageArray) throws IOException {
        this.BWImage = ImageIO.read(new ByteArrayInputStream(BWImageArray));
    }

    public CropImage(BufferedImage newBWImage) {
        this.BWImage = newBWImage;
    }

    public BufferedImage cropBWImage() {
        int topIndex = 0;
        int botIndex = 0;
        int leftIndex = 0;
        int rightIndex = 0;
        for (int i = 0; i < BWImage.getHeight(); ++i) {
            boolean found = false;
            for (int j = 0; j < BWImage.getWidth(); ++j) {
                if (BWImage.getRGB(j, i) == Color.BLACK.getRGB()) {
                    found = true;
                }
            }
            if (found) {
                topIndex = i;
                break;
            }
        }
        for (int i = BWImage.getHeight() - 1; i >= 0; --i) {
            boolean found = false;
            for (int j = 0; j < BWImage.getWidth(); ++j) {
                if (BWImage.getRGB(j, i) == Color.BLACK.getRGB()) {
                    found = true;
                }
            }
            if (found) {
                botIndex = i;
                break;
            }
        }
        for (int i = 0; i < BWImage.getWidth(); ++i) {
            boolean found = false;
            for (int j = topIndex; j < botIndex; ++j) {
                if (BWImage.getRGB(i, j) == Color.BLACK.getRGB()) {
                    found = true;
                }
            }
            if (found) {
                leftIndex = i;
                break;
            }
        }
        for (int i = BWImage.getWidth() - 1; i >= leftIndex; --i) {
            boolean found = false;
            for (int j = topIndex; j < botIndex; ++j) {
                if (BWImage.getRGB(i, j) == Color.BLACK.getRGB()) {
                    found = true;
                    break;
                }
            }
            if (found) {
                rightIndex = i;
                break;
            }
        }
        if (rightIndex - leftIndex == 0 || botIndex - topIndex == 0) {
            return null;
        }
        return BWImage.getSubimage(leftIndex, topIndex, rightIndex - leftIndex, botIndex - topIndex);
    }
}
