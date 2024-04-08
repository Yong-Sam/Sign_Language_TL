using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine.UI;

public class PythonConnect : MonoBehaviour
{
    public string serverAddress; // Python 서버 IP 주소
    public int serverPort = 50000; // Python 서버 포트 번호
    public RawImage displayRenderer; // 텍스처를 출력할 오브젝트의 렌더러

    private Texture2D receivedTexture;
    private TcpClient client;
    private Thread receiveThread;
    
    private object lockObject = new object();

    void Start()
    {
        Debug.Log("Start시작");
        ConnectToServer();
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.Start();
        Debug.Log("Start클래스 통과");
    }

    void ConnectToServer()
    {
        Debug.Log("ConnectToServer시작");
        client = new TcpClient(serverAddress, serverPort);
        Debug.Log("서버연결클래스 통과");
    }

    /*void ReceiveData()
    {
        byte[] buffer = new byte[4096];
        while (true)
        {
            using (NetworkStream stream = client.GetStream())
            {
                int bytesRead = stream.Read(buffer, 0, buffer.Length);
                if (bytesRead > 0)
                {
                    // 받은 데이터를 텍스처로 변환
                    receivedTexture = new Texture2D(2, 2);
                    receivedTexture.LoadImage(buffer);
                }
            }
        }
    }*/
    void ReceiveData()
    {
        Debug.Log("ReceiveData시작");
        try
        {
            NetworkStream stream = client.GetStream();
            byte[] buffer = new byte[4096];
            List<byte> receivedBytes = new List<byte>();
            byte[] jpegData = receivedBytes.ToArray();

            while (true)
            {
                int bytesRead = stream.Read(buffer, 0, buffer.Length);
                if (bytesRead == 0)
                {
                    // 연결이 끊어진 경우
                    break;
                }

                // 받은 데이터를 리스트에 추가
                receivedBytes.AddRange(buffer.Take(bytesRead));

                // 받은 데이터 처리
                receivedTexture = new Texture2D(2, 2);
                receivedTexture.LoadImage(jpegData);
                receivedTexture.Reinitialize(receivedTexture.width, receivedTexture.height);
            }
            
            lock (lockObject)
            {
                receivedTexture = new Texture2D(2, 2);
                receivedTexture.LoadImage(jpegData);
                receivedTexture.Reinitialize(receivedTexture.width, receivedTexture.height);
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"Error receiving data: {ex.Message}");
        }
        finally
        {
            client.Close();
        }
        Debug.Log("Receive Data클래스 통과");
    }
    
    void Update()
    {
        Debug.Log("Update시작");
        // 받은 텍스처를 렌더러에 적용하여 출력
        if (receivedTexture != null)
        {
            displayRenderer.texture = receivedTexture;
        }
        lock (lockObject)
        {
            if (receivedTexture != null)
            {
                displayRenderer.texture = receivedTexture;
            }
        }
        Debug.Log("Update클래스 통과");
    }

    void OnDestroy()
    {
        if (client != null)
        {
            client.Close();
        }
        if (receiveThread != null && receiveThread.IsAlive)
        {
            receiveThread.Abort();
        }
        Debug.Log("OnDestroy클래스 통과");
    }
}