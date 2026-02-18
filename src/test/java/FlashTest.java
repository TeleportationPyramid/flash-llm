
import io.github.teleportationpyramid.flash.*;

import java.io.IOException;

public class FlashTest {
    public static void main(String[] args) {

        // 測試 1: 載入 native library
        NativeLibraryLoader.getLibrary();
        System.out.println("✓ Native library 載入成功");

        // 測試 2: 檢測 GPU
//        int count = CudaDevice.getCount();
//        System.out.println("✓ 偵測到 " + count + " 個 GPU");

        try (var device = new CudaDevice(0);
             var blas = new CudaBlas(device)) {

            float[] a = {1, 2, 3, 4};
            float[] b = {5, 6, 7, 8};

            CudaTensor tensorA = CudaTensor.fromFloat(device, a, Precision.FP32);
            CudaTensor tensorB = CudaTensor.fromFloat(device, b, Precision.FP32);
            CudaTensor tensorC = CudaTensor.allocate(device, 4, Precision.FP32);

            blas.gemm(2, 2, 2, 1.0, tensorA, tensorB, 0.0, tensorC);

            float[] result = tensorC.toFloatArray();
            System.out.println("✓ 矩陣乘法成功，結果: " + java.util.Arrays.toString(result));
        }
    }
}
