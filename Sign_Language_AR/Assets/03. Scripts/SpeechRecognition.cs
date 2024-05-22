using UnityEngine;
using UnityEngine.UI;
using Microsoft.CognitiveServices.Speech;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using TMPro;

public class SpeechRecognition : MonoBehaviour
{
    private string subscriptionKey = "9c5d0d737b9148f6a60651b99d67aa12"; //YOUR_Subscription_KEY
    private string region = "koreacentral";  //YOUR_SERVICE_REGION

    private SpeechRecognizer recognizer;
    public Button speakButton;
    public TextMeshProUGUI btnText;
    public GameObject[] objectsToActivate;
    
    public TextMeshProUGUI displayText;

    void Start()
    {
        var config = SpeechConfig.FromSubscription(subscriptionKey, region);
        config.SpeechRecognitionLanguage = "ko-KR";
        recognizer = new SpeechRecognizer(config);

        recognizer.Recognized += (s, e) =>
        {
            if (e.Result.Reason == ResultReason.RecognizedSpeech)
            {
                Debug.Log($"Recognized: {e.Result.Text}");
                HandleRecognizedText(e.Result.Text);
            }
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
        btnText.text = "Stop Speaking";
    }

    async void StopRecognition()
    {
        Debug.Log("StopRecognition called");
        await recognizer.StopContinuousRecognitionAsync();
        speakButton.onClick.RemoveAllListeners();
        speakButton.onClick.AddListener(StartRecognition);
        btnText.text = "Start Speaking";
    }

    private void HandleRecognizedText(string text)
    {
        Debug.Log($"Handling recognized text: {text}");

        displayText.text = text;
        Debug.Log("자막 업데이트");

        if (objectsToActivate == null || objectsToActivate.Length == 0)
        {
            Debug.LogError("objectsToActivate 배열이 초기화되지 않았습니다.");
            return;
        }
        
        // 모든 오브젝트를 비활성화
        foreach (GameObject obj in objectsToActivate)
        {
            if (obj != null)
            {
                Debug.Log($"Deactivating object: {obj.name}");
                obj.SetActive(false);
            }
            else
            {
                Debug.LogWarning("objectsToActivate 배열에 null 값이 포함되어 있습니다.");
            }
        }

        // 특정 텍스트에 따라 오브젝트 활성화
        if (text.Contains("안녕하세요"))
        {
            Debug.Log("Activate 안녕하세요");
            if (objectsToActivate.Length > 0)
            {
                objectsToActivate[0].SetActive(true);
            }
        }
        else if (text.Contains("만나서"))
        {
            Debug.Log("Activate 만나서");
            if (objectsToActivate.Length > 1)
            {
                objectsToActivate[1].SetActive(true);
            }
        }
        // 추가 조건에 따라 오브젝트를 활성화할 수 있습니다.
    }

    private void OnDestroy()
    {
        recognizer.StopContinuousRecognitionAsync().Wait();
        recognizer.Dispose();
    }
}
