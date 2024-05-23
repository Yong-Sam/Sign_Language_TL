using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UIElements;

public class Intro : MonoBehaviour
{
    public void OnClickStart()
    {
        Debug.Log("시작하기");
        SceneManager.LoadScene("SL_TL");
    }
}
