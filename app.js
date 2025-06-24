// --- HTML 元素獲取 ---
const uploadButton = document.getElementById('upload-button');
const imageUpload = document.getElementById('image-upload');
const imagePreview = document.getElementById('image-preview');
const uploadPrompt = document.getElementById('upload-prompt');
const resultBox = document.getElementById('result-box');

// --- 主函數入口 ---
async function main() {
    // 點擊按鈕時，觸發檔案選擇框
    uploadButton.addEventListener('click', () => {
        imageUpload.click();
    });

    // 當用戶選擇了檔案時
    imageUpload.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                // 顯示圖片預覽
                imagePreview.src = e.target.result;
                imagePreview.classList.remove('hidden');
                uploadPrompt.classList.add('hidden');
                // 圖片載入後，進行分析
                imagePreview.onload = () => {
                    runModel(imagePreview);
                };
            };
            reader.readAsDataURL(file);
        }
    });
}

// --- 模型運行函數 ---
async function runModel(imageElement) {
    resultBox.innerText = '正在載入模型...';
    resultBox.className = 'result-box-placeholder';

    try {
        // 1. 創建 ONNX Runtime 推理會話
        //    確保 PIMNet.onnx 檔案與 index.html 在同一目錄下
        const session = await ort.InferenceSession.create('./PIMNet.onnx');
        resultBox.innerText = '模型載入成功，正在分析圖片...';

        // 2. 圖像預處理
        const imageTensor = await preprocessImage(imageElement);

        // 3. 準備模型的輸入
        //    您需要知道在匯出ONNX時定義的輸入名，默認為 'input'
        const feeds = { 'input': imageTensor };

        // 4. 運行模型
        const results = await session.run(feeds);

        // 5. 處理輸出結果
        //    您需要知道匯出ONNX時定義的輸出名，默認為 'output'
        const outputData = results.output.data;
        
        // 假設模型輸出一個機率值, >0.5為真實, 否則為偽造
        const score = outputData[0];
        
        if (score > 0.5) {
            resultBox.innerText = `真實人臉 (置信度: ${score.toFixed(2)})`;
            resultBox.className = 'result-real';
        } else {
            resultBox.innerText = `虛假人臉 (置信度: ${(1 - score).toFixed(2)})`;
            resultBox.className = 'result-fake';
        }

    } catch (e) {
        console.error(e);
        resultBox.innerText = `出現錯誤: ${e.message}`;
        resultBox.className = 'result-fake';
    }
}

// --- 圖像預處理函數 ---
async function preprocessImage(imageElement) {
    // 創建一個臨時的 canvas 來處理圖像
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const modelWidth = 224; // 您的模型需要的輸入寬度
    const modelHeight = 224; // 您的模型需要的輸入高度
    canvas.width = modelWidth;
    canvas.height = modelHeight;

    // 將圖像繪製到 canvas 上，並縮放到模型所需尺寸
    ctx.drawImage(imageElement, 0, 0, modelWidth, modelHeight);
    
    // 從 canvas 獲取圖像數據
    const imageData = ctx.getImageData(0, 0, modelWidth, modelHeight);
    const data = imageData.data;

    // 將像素數據從 [0, 255] 轉換到 [0, 1] 的浮點數，並進行 NCHW 排列
    // ONNX Runtime 需要的格式通常是 [batch_size, channels, height, width]
    const red = [], green = [], blue = [];
    for (let i = 0; i < data.length; i += 4) {
        red.push(data[i] / 255);
        green.push(data[i + 1] / 255);
        blue.push(data[i + 2] / 255);
    }
    // 按照 [RRR...GGG...BBB] 的順序排列
    const float32Data = new Float32Array([...red, ...green, ...blue]);

    // 創建 ONNX Runtime 張量
    const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, modelHeight, modelWidth]);
    return inputTensor;
}

// 啟動程序
main();