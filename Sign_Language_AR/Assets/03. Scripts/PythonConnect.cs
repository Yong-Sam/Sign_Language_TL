using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine.UI;

public class PythonConnect : MonoBehaviour
{
    public string serverAddress = "192.168.0.27"; // Python 서버 IP 주소
    public int serverPort = 5052; // Python 서버 포트 번호
    public RawImage displayRenderer; // 텍스처를 출력할 오브젝트의 렌더러

    private Texture2D receivedTexture;
    private TcpClient client;
    private Thread receiveThread;

    void Start()
    {
        ConnectToServer();
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.Start();
    }

    void ConnectToServer()
    {
        client = new TcpClient(serverAddress, serverPort);
    }

    void ReceiveData()
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
    }

    void Update()
    {
        // 받은 텍스처를 렌더러에 적용하여 출력
        if (receivedTexture != null)
        {
            displayRenderer.texture = receivedTexture;
        }
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
    }
}