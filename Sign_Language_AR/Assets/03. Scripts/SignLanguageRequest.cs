using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

public class SignLanguageRequest : MonoBehaviour
{
    void Start()
    {
        StartCoroutine(GetSignLanguageCode());
    }

    IEnumerator GetSignLanguageCode()
    {
        string url = "http://127.0.0.1:8000/get-latest-sign-language-code";

        using (UnityWebRequest webRequest = UnityWebRequest.Get(url))
        {
            yield return webRequest.SendWebRequest();

            if (webRequest.result == UnityWebRequest.Result.ConnectionError ||
                webRequest.result == UnityWebRequest.Result.ProtocolError)
            {
                Debug.LogError("Error: " + webRequest.error);
            }
            else
            {
                // Parse the JSON response
                SignLanguageData data = JsonUtility.FromJson<SignLanguageData>(webRequest.downloadHandler.text);
                Debug.Log("Latest Sign Language Code: " + data.latest_code);
            }
        }
    }
}