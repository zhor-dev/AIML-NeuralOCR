package ImageProcessing;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;

public class GetBlackAndWhiteImage {

    private BufferedImage image;

    public GetBlackAndWhiteImage(byte []imageArray) throws IOException {
        this.image = ImageIO.read(new ByteArrayInputStream(imageArray));
    }

    public GetBlackAndWhiteImage(BufferedImage i) {
        this.image = i;
    }

    public BufferedImage changeToBlackWhite(int threshold) {
        BufferedImage blackWhiteImage = new BufferedImage(image.getWidth(), image.getHeight(), image.getType());
        boolean isContainBlack = false;
        for (int i = 0; i < blackWhiteImage.getHeight(); ++i) {
            for (int j = 0; j < blackWhiteImage.getWidth(); ++j) {
                Color c = new Color(image.getRGB(j , i));
                double avg = (c.getRed() + c.getGreen() + c.getBlue()) / 3.0;
                if (avg < threshold) {
                    blackWhiteImage.setRGB(j, i, Color.BLACK.getRGB());
                    isContainBlack = true;
                } else {
                    blackWhiteImage.setRGB(j, i, Color.WHITE.getRGB());
                }
            }
        }
        if (!isContainBlack) {
            for (int i = 0; i < blackWhiteImage.getHeight(); ++i) {
                for (int j = 0; j < blackWhiteImage.getWidth(); ++j) {
                    if (threshold < 200) {
                        blackWhiteImage.setRGB(j, i, Color.BLACK.getRGB());
                    }
                }
            }
        }
        return blackWhiteImage;
    }
}
