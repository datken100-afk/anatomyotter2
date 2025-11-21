
import { GoogleGenAI, Type, Schema, GenerateContentResponse } from "@google/genai";
import { Difficulty, GeneratedMCQResponse, GeneratedStationResponse, MentorResponse, StationItem } from "../types";

// Lấy API Key từ biến môi trường (Vercel Environment Variable)
const apiKey = process.env.API_KEY || '';

// Initialize Gemini Client
const ai = new GoogleGenAI({ apiKey });

// UPGRADE: Use Gemini 3 Pro for superior reasoning
const MODEL_MCQ = "gemini-3-pro-preview";
const MODEL_VISION = "gemini-2.5-flash"; // Updated from 1.5 to 2.5
const MODEL_CHAT = "gemini-2.5-flash";

interface ContentFile {
    content: string;
    isText: boolean;
}

// Token Limits
const LIMIT_THEORY_CHARS = 2400000; 
const LIMIT_CLINICAL_CHARS = 1000000; 
const LIMIT_SAMPLE_CHARS = 200000; 

// --- RETRY LOGIC HELPER ---
const wait = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

async function retryGeminiCall<T>(
  call: () => Promise<T>,
  retries: number = 3,
  initialDelay: number = 2000
): Promise<T> {
  let lastError: any;
  
  for (let i = 0; i < retries; i++) {
    try {
      return await call();
    } catch (error: any) {
      lastError = error;
      
      // Check for common API errors
      const isRateLimit = 
        error.status === 429 || 
        error.status === 503 ||
        (error.message && (
          error.message.includes("429") || 
          error.message.includes("quota") || 
          error.message.includes("RESOURCE_EXHAUSTED") ||
          error.message.includes("Overloaded")
        ));

      // Check for Model Not Found (404) - Usually due to old code or region lock
      if (error.status === 404 || (error.message && error.message.includes("not found"))) {
          console.error("Model Not Found Error. Please check if you are using the latest code and a valid API Key.");
          throw new Error(`Lỗi Model AI (${error.status}): Không tìm thấy Model. Vui lòng Redeploy code mới nhất lên Vercel.`);
      }

      if (isRateLimit) {
        if (i === retries - 1) break; 
        console.warn(`Gemini Rate Limit hit. Retrying in ${initialDelay}ms... (Attempt ${i + 1}/${retries})`);
        await wait(initialDelay);
        initialDelay *= 2; 
      } else {
        throw error; 
      }
    }
  }
  
  const cleanMsg = lastError?.message || "Unknown error";
  if (cleanMsg.includes("quota") || cleanMsg.includes("RESOURCE_EXHAUSTED")) {
      throw new Error("Đã hết hạn mức sử dụng AI (Quota Exceeded). Vui lòng kiểm tra gói cước hoặc thử lại vào ngày mai.");
  }
  throw new Error(`Lỗi kết nối AI: ${cleanMsg}`);
}

export const generateMCQQuestions = async (
  topic: string,
  count: number,
  difficulties: Difficulty[],
  files: { theory?: ContentFile[]; clinical?: ContentFile[]; sample?: ContentFile[] } = {}
): Promise<GeneratedMCQResponse> => {
  if (!apiKey) throw new Error("Chưa cấu hình API Key. Vui lòng thêm API_KEY vào Vercel Environment Variables.");

  let systemInstruction = `
    Bạn là một giáo sư Y khoa hàng đầu. Nhiệm vụ của bạn là tạo đề thi trắc nghiệm giải phẫu học chất lượng cao.
    
    QUY TẮC PHÂN TÍCH TÀI LIỆU (TUÂN THỦ TUYỆT ĐỐI):
    1. DỮ LIỆU LÝ THUYẾT (Theory): CHỈ được sử dụng để tạo các câu hỏi thuộc mức độ: 
       - ${Difficulty.REMEMBER} (Ghi nhớ)
       - ${Difficulty.UNDERSTAND} (Hiểu)
       - ${Difficulty.APPLY} (Vận dụng thấp)

    2. DỮ LIỆU LÂM SÀNG (Clinical): CHỈ được sử dụng để tạo câu hỏi mức độ:
       - ${Difficulty.CLINICAL} (Lâm sàng/Ca bệnh)
       Câu hỏi lâm sàng bắt buộc phải là các Case Study (tình huống bệnh nhân) cụ thể.

    3. ĐỀ THI MẪU: Nếu có, hãy học phong cách đặt câu hỏi và format từ đó.

    CẤU TRÚC ĐỀ THI:
    - Tổng số câu: ${count} câu.
    - Chủ đề: "${topic}".
    - Các mức độ khó yêu cầu: ${difficulties.join(', ')}.
    - Mỗi câu hỏi có 4 lựa chọn, 1 đáp án đúng.
    - Giải thích: Phải cực kỳ chi tiết, trích dẫn lý do tại sao đúng/sai.
  `;

  const schema: Schema = {
    type: Type.OBJECT,
    properties: {
      questions: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            question: { type: Type.STRING },
            options: { type: Type.ARRAY, items: { type: Type.STRING } },
            correctAnswer: { type: Type.STRING },
            explanation: { type: Type.STRING },
            difficulty: { type: Type.STRING },
          },
          required: ["question", "options", "correctAnswer", "explanation", "difficulty"],
        },
      },
    },
    required: ["questions"],
  };

  const parts: any[] = [];

  const addContentParts = (fileItems: ContentFile[] | undefined, sectionTitle: string, usageInstruction: string, charLimit: number) => {
    if (!fileItems || fileItems.length === 0) return;

    parts.push({ text: `\n=== BẮT ĐẦU PHẦN: ${sectionTitle} ===\nCHỈ DẪN: ${usageInstruction}\n` });
    
    let currentChars = 0;

    for (const item of fileItems) {
        if (currentChars >= charLimit) {
             parts.push({ text: `\n[CẢNH BÁO: Đã ngưng tải thêm tài liệu phần này do vượt quá giới hạn bộ nhớ cho phép]\n` });
             break;
        }

        if (item.content) {
            if (item.isText) {
                let textToAdd = item.content;
                const remaining = charLimit - currentChars;

                if (textToAdd.length > remaining) {
                    textToAdd = textToAdd.substring(0, remaining) + "\n\n[...Nội dung file này đã bị cắt bớt do giới hạn bộ nhớ AI...]";
                }
                
                parts.push({ text: `\n--- FILE CONTENT ---\n${textToAdd}\n` });
                currentChars += textToAdd.length;
            } else {
                const base64Data = item.content.includes('base64,') ? item.content.split('base64,')[1] : item.content;
                parts.push({
                    inlineData: {
                        mimeType: "application/pdf", 
                        data: base64Data
                    }
                });
                currentChars += 50000; 
            }
        }
    }
    parts.push({ text: `=== KẾT THÚC PHẦN: ${sectionTitle} ===\n` });
  };

  addContentParts(files.theory, "TÀI LIỆU LÝ THUYẾT", `Dùng cho câu hỏi mức độ thấp.`, LIMIT_THEORY_CHARS);
  addContentParts(files.clinical, "TÀI LIỆU LÂM SÀNG", `CHỈ Dùng cho câu hỏi mức độ ${Difficulty.CLINICAL}.`, LIMIT_CLINICAL_CHARS);
  addContentParts(files.sample, "ĐỀ THI MẪU", "Tham khảo cách đặt câu hỏi.", LIMIT_SAMPLE_CHARS);

  parts.push({ text: `Hãy "Suy nghĩ" (Thinking) kỹ về phân phối câu hỏi, sau đó soạn thảo ${count} câu hỏi trắc nghiệm về chủ đề "${topic}" theo đúng định dạng JSON đã yêu cầu.` });

  try {
    console.log(`Generating MCQs with model: ${MODEL_MCQ}`);
    const response = await retryGeminiCall<GenerateContentResponse>(() => ai.models.generateContent({
      model: MODEL_MCQ,
      contents: { parts: parts },
      config: {
        systemInstruction: systemInstruction,
        responseMimeType: "application/json",
        responseSchema: schema,
        thinkingConfig: { thinkingBudget: 2048 }, 
      },
    }));

    let text = response.text;
    if (!text) throw new Error("No response from AI");
    
    const jsonBlockMatch = text.match(/```json\s*([\s\S]*?)\s*```/);
    if (jsonBlockMatch) {
        text = jsonBlockMatch[1];
    } else {
        text = text.replace(/```json/g, '').replace(/```/g, '');
    }
    
    text = text.trim();
    const parsed = JSON.parse(text);

    if (!parsed || typeof parsed !== 'object' || !Array.isArray(parsed.questions)) {
       throw new Error("Invalid response structure");
    }

    return parsed as GeneratedMCQResponse;

  } catch (error: any) {
    console.error("Gemini API Error:", error);
    if (error.message && (error.message.includes("quá tải") || error.message.includes("hết hạn mức") || error.message.includes("Redeploy"))) {
        throw error;
    }
    if (error.message && error.message.includes("token count exceeds")) {
        throw new Error("Tổng dung lượng tài liệu quá lớn. Vui lòng bớt file.");
    }
    throw error;
  }
};

// --- Generate Spot Test Question from Image (Vision) ---
export interface StationQuestionResponse {
    isValid: boolean;
    questions?: {
        questionText: string;
        correctAnswer: string;
        explanation: string;
    }[];
}

export const generateStationQuestionFromImage = async (base64Image: string, topic?: string): Promise<StationQuestionResponse> => {
    if (!apiKey) throw new Error("Chưa cấu hình API Key.");
    
    const systemInstruction = `
    Bạn là giám khảo thi chạy trạm (Spot Test) Giải phẫu học cực kỳ nghiêm túc.
    
    NHIỆM VỤ 1: KIỂM TRA TÍNH HỢP LỆ & ĐÚNG CHỦ ĐỀ: "${topic || 'Giải phẫu học'}".
    - Hình ảnh HỢP LỆ: Hình giải phẫu rõ ràng, có chú thích/leader lines, ĐÚNG CHỦ ĐỀ.
    - Hình ảnh KHÔNG HỢP LỆ: Toàn chữ, Mục lục, Sai chủ đề.

    NHIỆM VỤ 2: RA ĐỀ (Nếu Hợp lệ):
    1. Chọn MỘT cấu trúc giải phẫu quan trọng nhất trong hình LIÊN QUAN ĐẾN CHỦ ĐỀ.
    2. Đặt câu hỏi định danh trực tiếp (VD: "Chi tiết số 1 là gì?").
    3. Đáp án Tiếng Việt chính xác.

    Output JSON format: { "isValid": boolean, "questions": [...] }
    `;

    const prompt = topic 
        ? `Kiểm tra xem hình này có chứa cấu trúc giải phẫu thuộc chủ đề "${topic}" không. Nếu có, hãy tạo 1 câu hỏi trạm.` 
        : "Kiểm tra xem đây có phải là hình giải phẫu hợp lệ không. Nếu có, hãy tạo 1 câu hỏi trạm.";

    try {
        const cleanBase64 = base64Image.includes('base64,') ? base64Image.split('base64,')[1] : base64Image;
        
        console.log(`Generating Station with model: ${MODEL_VISION}`);
        const response = await retryGeminiCall<GenerateContentResponse>(() => ai.models.generateContent({
            model: MODEL_VISION,
            contents: { 
                role: 'user', 
                parts: [
                    { text: prompt },
                    { inlineData: { mimeType: 'image/jpeg', data: cleanBase64 } }
                ] 
            },
            config: {
                systemInstruction: systemInstruction,
                responseMimeType: "application/json",
                responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                        isValid: { type: Type.BOOLEAN },
                        questions: {
                            type: Type.ARRAY,
                            items: {
                                type: Type.OBJECT,
                                properties: {
                                    questionText: { type: Type.STRING },
                                    correctAnswer: { type: Type.STRING },
                                    explanation: { type: Type.STRING }
                                },
                                required: ["questionText", "correctAnswer", "explanation"]
                            }
                        }
                    },
                    required: ["isValid"]
                }
            }
        }));

        let text = response.text || "";
        text = text.replace(/```json/g, '').replace(/```/g, '').trim();
        return JSON.parse(text) as StationQuestionResponse;
    } catch (e: any) {
        console.error("Vision API Error", e);
        if (e.message && (e.message.includes("quá tải") || e.message.includes("quota") || e.message.includes("429") || e.message.includes("Redeploy"))) {
            throw e;
        }
        return { isValid: false, questions: [] };
    }
};

export const analyzeResultWithOtter = async (
    topic: string,
    stats: Record<string, { correct: number, total: number }>
): Promise<MentorResponse> => {
    if (!apiKey) return { analysis: "Chưa có API Key", strengths: [], weaknesses: [], roadmap: [] };

    const statsDescription = Object.entries(stats)
        .map(([diff, val]) => {
             const pct = val.total > 0 ? Math.round((val.correct / val.total) * 100) : 0;
             return `- ${diff}: ${val.correct}/${val.total} câu (${pct}%)`;
        })
        .join('\n');

    const prompt = `
    Đóng vai là "Rái cá nhỏ" 🦦 - gia sư AI giải phẫu.
    Học viên vừa làm bài thi chủ đề: "${topic}".
    DỮ LIỆU: \n${statsDescription}
    
    NHIỆM VỤ:
    1. Phân tích năng lực.
    2. Chỉ ra Điểm mạnh/Yếu.
    3. Lộ trình cải thiện (4 bước cụ thể, kỹ thuật học tập rõ ràng).
    
    JSON Output: { "analysis": string, "strengths": string[], "weaknesses": string[], "roadmap": [{ "step": string, "details": string }] }
    `;

    try {
        console.log(`Analyzing with model: ${MODEL_MCQ}`);
        const response = await retryGeminiCall<GenerateContentResponse>(() => ai.models.generateContent({
            model: MODEL_MCQ,
            contents: { role: 'user', parts: [{ text: prompt }] },
            config: {
                responseMimeType: "application/json",
                thinkingConfig: { thinkingBudget: 2048 }
            }
        }));

        let text = response.text || "";
        text = text.replace(/```json/g, '').replace(/```/g, '').trim();
        return JSON.parse(text) as MentorResponse;
    } catch (e) {
        console.error(e);
        return {
            analysis: "Úi cha! Rái cá đang bận bắt cá nên không phân tích được rồi. Thử lại sau nhé! 🦦",
            strengths: [],
            weaknesses: [],
            roadmap: []
        };
    }
};

export const chatWithOtter = async (history: {role: 'user' | 'model', text: string, image?: string}[], message: string, image?: string): Promise<string> => {
    if (!apiKey) return "Vui lòng nhập API Key để chat với Rái cá!";

    const systemInstruction = `Bạn là "Rái cá nhỏ" (Little Otter) 🦦 - trợ lý ảo GIẢI PHẪU HỌC.
    - Vui vẻ, chuyên nghiệp, dùng emoji 🦦 🦴 🧠.
    - Giải đáp kiến thức giải phẫu, phân tích hình ảnh.
    - Trình bày Markdown gọn gàng.
    `;

    const contents = history.map(msg => {
        const parts: any[] = [{ text: msg.text }];
        if (msg.image) {
             try {
                 const base64Data = msg.image.includes('base64,') ? msg.image.split('base64,')[1] : msg.image;
                 const mimeType = msg.image.match(/data:([^;]+);base64,/)?.[1] || 'image/jpeg';
                 parts.push({ inlineData: { mimeType, data: base64Data }});
             } catch (e) { console.warn("History image error", e); }
        }
        return { role: msg.role, parts };
    });

    const currentParts: any[] = [{ text: message }];
    if (image) {
        try {
            const base64Data = image.includes('base64,') ? image.split('base64,')[1] : image;
            const mimeType = image.match(/data:([^;]+);base64,/)?.[1] || 'image/jpeg';
            currentParts.push({ inlineData: { mimeType, data: base64Data }});
        } catch (e) { console.warn("Current image error", e); }
    }
    contents.push({ role: 'user', parts: currentParts });

    try {
        console.log(`Chatting with model: ${MODEL_CHAT}`);
        const response = await retryGeminiCall<GenerateContentResponse>(() => ai.models.generateContent({
            model: MODEL_CHAT,
            contents,
            config: { systemInstruction }
        }));
        return response.text || "Rái cá đang bơi đi đâu mất rồi... 🦦";
    } catch (e) {
        console.error(e);
        return "Úi! Mạng bị nghẽn hoặc lỗi kết nối. Bạn hỏi lại nhé? 🦦";
    }
};
