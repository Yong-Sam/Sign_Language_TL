using System;
using UnityEngine;
using System.Collections;
using System.Net;
using System.IO;
using TMPro;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine.UI;

public class SL_HttpServer : MonoBehaviour
{
    public TextMeshProUGUI displayText;
    private HashSet<string> currentWords = new HashSet<string>();
    private List<string> currentSentence = new List<string>();
    private HttpListener listener;
    private string url = "http://localhost:5000/";
    private int port = 5000;

    void Start()
    {
        listener = new HttpListener();
        listener.Prefixes.Add(url);
        listener.Start();
        StartListening();
        Debug.Log("Server running on port " + port);
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
                    
                    // 단어 번호에 따라 Text 컴포넌트에 출력 (메인 스레드에서 실행)
                    UnityMainThreadDispatcher.Instance().Enqueue(() =>
                    {
                        string word = GetWordFromNumber(wordData.word_number);
                        if (!string.IsNullOrEmpty(word))
                        {
                            if (!currentWords.Contains(word))
                            {
                                if (currentSentence.Count >= 3)
                                {
                                    currentSentence.Clear();
                                    currentWords.Clear();
                                }
                                currentSentence.Add(word);
                                currentWords.Add(word);
                                displayText.text = string.Join(" ", currentSentence);
                            }
                        }
                    });
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

    private string GetWordFromNumber(int number)
    {
        switch (number)
        {
            case 1:
                return "안녕하세요.";
            case 2:
                return "만나서";
            case 3:
                return "반갑습니다.";
            case 4:
                return "저(의/는)";
            case 5:
                return "이름";
            default:
                return null;
        }
    }
}
