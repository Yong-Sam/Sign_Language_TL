using System;
using UnityEngine;
using System.Collections;
using System.Net;
using System.IO;
using TMPro;
using UnityEngine.UI;

public class SL_HttpServer : MonoBehaviour
{
    public TextMeshProUGUI displayText;
    private HttpListener listener;
    private string url = "http://localhost:5000/";
    private int port = 5000;

    void Start()
    {
        listener = new HttpListener();
        listener.Prefixes.Add(url);
        listener.Start();
        StartListening();
    }

    void StartListening()
    {
        listener.BeginGetContext(new AsyncCallback(ListenerCallback), listener);
    }

    void ListenerCallback(IAsyncResult result)
    {
        if (listener.IsListening)
        {
            HttpListenerContext context = listener.EndGetContext(result);
            HandleRequest(context);
            StartListening();
        }
    }

    [Serializable]
    public class WordData
    {
        public int word_number;
    }
    void HandleRequest(HttpListenerContext context)
    {
        HttpListenerRequest request = context.Request;
        HttpListenerResponse response = context.Response;

        if (request.HttpMethod == "POST" && request.RawUrl == "/receive_word")
        {
            try
            {
                using (StreamReader reader = new StreamReader(request.InputStream))
                {
                    string jsonData = reader.ReadToEnd();
                    WordData wordData = JsonUtility.FromJson<WordData>(jsonData);
                    
                    // 단어 번호에 따라 Text 컴포넌트에 출력
                    switch (wordData.word_number)
                    {
                        case 1:
                            displayText.text = "안녕하세요.";
                            break;
                        case 2:
                            displayText.text = "만나서"; 
                            break;
                        case 3:
                            displayText.text = "반갑습니다.";
                            break;
                        case 4:
                            displayText.text = "저(의/는)";
                            break;
                        case 5:
                            displayText.text = "이름";
                            break;
                        // 나머지 단어도 추가...
                        default:
                            displayText.text = "(수화 인식 중..)";
                            break;
                    }
                }
            }
            catch (System.Exception e)
            {
                Debug.Log("Exception: " + e.Message);
            }

            // 응답 처리
            byte[] buffer = System.Text.Encoding.UTF8.GetBytes("OK");
            response.ContentLength64 = buffer.Length;
            Stream output = response.OutputStream;
            output.Write(buffer, 0, buffer.Length);
            output.Close();
        }
    }
}