using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Microsoft.CognitiveServices.Speech;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using TMPro;
using UnityEngine.Android;
using UnityEngine.Video;

public class SpeechRecognition : MonoBehaviour
{
    private string subscriptionKey = "9c5d0d737b9148f6a60651b99d67aa12"; //YOUR_Subscription_KEY
    private string region = "koreacentral";  //YOUR_SERVICE_REGION

    private SpeechRecognizer recognizer;
    public Button speakButton;
    public Image targetImage;
    public Sprite Mic_On;
    public Sprite Mic_Off;
    private bool isMicON = true;
    
    public VideoPlayer videoPlayer; // 비디오 플레이어 참조
    public VideoClip[] videoClips; // 단어별 비디오 클립 배열
    public TextMeshProUGUI displayText;

    private HashSet<string> currentWords = new HashSet<string>();
    private List<string> currentSentence = new List<string>();

    void Start()
    {
        // Android에서 마이크 권한 요청
        if (Application.platform == RuntimePlatform.Android)
        {
            if (!Permission.HasUserAuthorizedPermission(Permission.Microphone))
            {
                Permission.RequestUserPermission(Permission.Microphone);
            }
        }
        
        var config = SpeechConfig.FromSubscription(subscriptionKey, region);
        config.SpeechRecognitionLanguage = "ko-KR";
        recognizer = new SpeechRecognizer(config);

        recognizer.Recognized += (s, e) =>
        {
            if (e.Result.Reason == ResultReason.RecognizedSpeech)
            {
                Debug.Log($"Recognized: {e.Result.Text}");
                UnityMainThreadDispatcher.Instance().Enqueue(() => HandleRecognizedText(e.Result.Text));            }
            else if (e.Result.Reason == ResultReason.NoMatch)
            {
                Debug.Log("No speech could be recognized.");
            }
        };

        recognizer.Canceled += (s, e) =>
        {
            Debug.Log($"Canceled: Reason={e.Reason.ToString()}, ErrorDetails={e.ErrorDetails}");
            recognizer.StopContinuousRecognitionAsync();
        };

        recognizer.SessionStopped += (s, e) =>
        {
            Debug.Log("Session stopped.");
            recognizer.StopContinuousRecognitionAsync();
        };

        speakButton.onClick.AddListener(StartRecognition);
    }

    async void StartRecognition()
    {
        Debug.Log("StartRecognition called");
        await recognizer.StartContinuousRecognitionAsync();
        speakButton.onClick.RemoveAllListeners();
        speakButton.onClick.AddListener(StopRecognition);
        ChangeImage();
        displayText.text = "";
        
        PlayVideo(1); // listening 비디오 재생 (끄덕임)
    }

    async void StopRecognition()
    {
        Debug.Log("StopRecognition called");
        
        PlayVideo(0);// basic 비디오 재생 (멈춤상태)
        
        await recognizer.StopContinuousRecognitionAsync();
        speakButton.onClick.RemoveAllListeners();
        speakButton.onClick.AddListener(StartRecognition);
        ChangeImage();
        
        displayText.text = "";
    }

    private void HandleRecognizedText(string text)
    {
        try
        {
            Debug.Log($"Handling recognized text: {text}");

            displayText.text = text;
            Debug.Log("자막 업데이트");

            
            // 띄어쓰기로 텍스트를 분할하여 단어 리스트에 추가
            string[] words = text.Split(' ');
    
            foreach (string word in words)
            {
                if (!string.IsNullOrEmpty(word) && !currentWords.Contains(word))
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

            // 특정 텍스트에 따라 영상 활성화
            if (text.Contains("안녕하세요") || text.Contains("안녕하십니까"))
            {
                Debug.Log("Activate 안녕하세요/안녕하십니까");
                PlayVideo(2); // 비디오 재생
            }
            else if (text.Contains("만나서"))
            {
                Debug.Log("Activate 만나서");
                PlayVideo(3); // 비디오 재생
            }
            else if (text.Contains("반갑습니다") || text.Contains("반가워요"))
            {
                Debug.Log("Activate 반갑습니다/반가워요");
                PlayVideo(4); // 비디오 재생
            }
            else if (text.Contains("당신"))
            {
                Debug.Log("Activate 당신(을/이)");
                PlayVideo(5); // 비디오 재생
            }
            else if (text.Contains("오늘"))
            {
                Debug.Log("Activate 오늘");
                PlayVideo(6); // 비디오 재생
            }
            else if (text.Contains("날씨"))
            {
                Debug.Log("Activate 날씨");
                PlayVideo(7); // 비디오 재생
            }
            else if (text.Contains("좋네요") || text.Contains("좋아요") || text.Contains("좋습니다"))
            {
                Debug.Log("Activate 좋네요/좋아요/좋습니다");
                PlayVideo(8); // 비디오 재생
            }
            else if (text.Contains("감사합니다") || text.Contains("감사해요") || text.Contains("고맙습니다") || text.Contains("고마워요"))
            {
                Debug.Log("Activate 감사합니다/감사해요/고맙습니다/고마워요");
                PlayVideo(9); // 비디오 재생
            }
            else if (text.Contains("죄송합니다") || text.Contains("죄송해요") || text.Contains("미안합니다") || text.Contains("미안해요"))
            {
                Debug.Log("Activate 죄송합니다/죄송해요/미안합니다/미안해요");
                PlayVideo(10); // 비디오 재생
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Exception in HandleRecognizedText: {e.Message}");
        }
    }

    private void PlayVideo(int index)
    {
        string path = System.IO.Path.Combine(Application.streamingAssetsPath, "basic_.mp4");
        videoPlayer.url = path;
        
        if (videoPlayer != null && videoClips != null && index >= 0 && index < videoClips.Length)
        {
            videoPlayer.clip = videoClips[index];
            videoPlayer.Play();
        }
        else
        {
            Debug.LogError("Invalid video index or video player not set.");
        }
    }

    private void ChangeImage()
    {
        if (isMicON)
        {
            targetImage.sprite = Mic_On;
        }
        else
        {
            targetImage.sprite = Mic_Off;
        }

        isMicON = !isMicON; //마이크 온오프 변경
    }
    
    private void OnDestroy()
    {
        recognizer.StopContinuousRecognitionAsync().Wait();
        recognizer.Dispose();
    }
}
