package ImageProcessing;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.ArrayList;

public class RecognizeImageParts {

    private BufferedImage cropedImage;

    public RecognizeImageParts(byte []BWImageArray) throws IOException {
        this.cropedImage = ImageIO.read(new ByteArrayInputStream(BWImageArray));
    }

    public RecognizeImageParts(BufferedImage CImg) {
        this.cropedImage = CImg;
    }

    public ArrayList<BufferedImage> getImageDetectedParts() {
        return clearEmptyPapers();
    }

    private ArrayList<BufferedImage> recognizeBWImageParts() {
        ArrayList<BufferedImage> tmpResult = new ArrayList<>();
        BufferedImage tmpImage = cropedImage;
        for (int i = 0; i < tmpImage.getWidth(); ++i) {
            boolean end = false;
            for (int j = 0; j < tmpImage.getHeight(); ++j) {
                if (tmpImage.getRGB(i, j) == Color.BLACK.getRGB()) {
                    end = true;
                    break;
                }
            }
            if (!end && i > 5) {
                tmpResult.add(tmpImage.getSubimage(0, 0, i, tmpImage.getHeight()));
                tmpImage = tmpImage.getSubimage(i, 0, tmpImage.getWidth() - i, tmpImage.getHeight());
                tmpImage = (new CropImage(tmpImage)).cropBWImage();
                i = 0;
            }
        }
        tmpResult.add(tmpImage);
        ArrayList<BufferedImage> result = new ArrayList<>();
        for (BufferedImage aTmpResult : tmpResult) {
            result.add((new CropImage(aTmpResult)).cropBWImage());
        }
        return result;
    }

    private ArrayList<BufferedImage> correctRectCharacters() {
        int k = 0;
        ArrayList<BufferedImage> result = new ArrayList<>();
        ArrayList<BufferedImage> tmpResult = recognizeBWImageParts();
        while (k < tmpResult.size()) {
            for (int i = 0; i < tmpResult.get(k).getHeight(); ++i) {
                for (int j = 0; j < tmpResult.get(k).getWidth(); ++j) {
                    int count = 0;
                    while ((j < tmpResult.get(k).getWidth())
                            && (tmpResult.get(k).getRGB(j, i) == Color.BLACK.getRGB())) {
                        ++count;
                        ++j;
                    }
                    if (count > 0 && count < 7) {
                        boolean clear = true;
                        int[] cnt = new int [7];
                        for (int q = 0; q < 7; ++q) {
                            cnt[q] = 0;
                        }
                        int p = 0;
                        int sz = j - count;
                        for (j = sz; j < sz + count; ++j) {
                            int t = i + 1;
                            while (j < tmpResult.get(k).getWidth() && t < tmpResult.get(k).getHeight() && t < i + 8
                                    && tmpResult.get(k).getRGB(j, t) == Color.BLACK.getRGB()) {
                                if (tmpResult.get(k).getRGB(j, t) == Color.BLACK.getRGB()) {
                                    ++cnt[p];
                                    ++t;
                                }
                            }
                            if (cnt[p] >= 7) {
                                clear = false;
                            }
                            ++p;
                        }
                        if (clear) {
                            int s = j - count;
                            for (int t = i; t < i + min(cnt) + 1; ++t) {
                                for (j = s; j < s + count; ++j) {
                                    if (j < tmpResult.get(k).getWidth() && t < tmpResult.get(k).getHeight())
                                        tmpResult.get(k).setRGB(j, t, Color.WHITE.getRGB());
                                }
                            }
                        }
                    }
                }
            }
            boolean is_clear = true;
            for (int i = 0; i < tmpResult.get(k).getHeight(); ++i) {
                for (int j = 0; j < tmpResult.get(k).getWidth(); ++j) {
                    if (tmpResult.get(k).getRGB(j, i) == Color.BLACK.getRGB()) {
                        is_clear = false;
                    }
                }
            }
            if (!is_clear) {
                result.add((new CropImage(tmpResult.get(k))).cropBWImage());
            } else {
                result.add(tmpResult.get(k));
            }
            ++k;
        }
        return result;
    }

    private ArrayList<BufferedImage> clearEmptyPapers() {
        ArrayList<BufferedImage> result = new ArrayList<>();
        ArrayList<BufferedImage> tmpResult = correctRectCharacters();
        int k = 0;
        while (k < tmpResult.size()) {
            int cnt = 0;
            if (tmpResult.get(k) != null) {
                for (int i = 0; i < tmpResult.get(k).getHeight(); ++i) {
                    for (int j = 0; j < tmpResult.get(k).getWidth(); ++j) {
                        if (tmpResult.get(k).getRGB(j, i) == Color.BLACK.getRGB()) {
                            ++cnt;
                        }
                    }
                }
                if (cnt > 10) {
                    result.add(tmpResult.get(k));
                }
            }
            ++k;
        }
        return result;
    }


    private int min(int... arr) {
        int min =  arr[0];
        for (int i = 1; i < arr.length; ++i) {
            if (arr[i] < min) {
                min = arr[i];
            }
        }
        return min;
    }
}
