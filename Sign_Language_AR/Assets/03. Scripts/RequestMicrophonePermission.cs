using UnityEngine;

public class RequestMicrophonePermission : MonoBehaviour
{
    void Start()
    {
        if (Application.platform == RuntimePlatform.Android)
        {
            RequestPermission();
        }
    }

    private void RequestPermission()
    {
        using (AndroidJavaClass unityPlayer = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
        {
            AndroidJavaObject currentActivity = unityPlayer.GetStatic<AndroidJavaObject>("currentActivity");
            using (AndroidJavaObject permissionService = new AndroidJavaObject("android.support.v4.content.ContextCompat"))
            {
                string permission = "android.permission.RECORD_AUDIO";
                int permissionCheck = permissionService.CallStatic<int>("checkSelfPermission", currentActivity, permission);
                
                if (permissionCheck != 0)
                {
                    using (AndroidJavaClass activityCompat = new AndroidJavaClass("android.support.v4.app.ActivityCompat"))
                    {
                        activityCompat.CallStatic("requestPermissions", currentActivity, new string[] { permission }, 0);
                    }
                }
            }
        }
    }
}