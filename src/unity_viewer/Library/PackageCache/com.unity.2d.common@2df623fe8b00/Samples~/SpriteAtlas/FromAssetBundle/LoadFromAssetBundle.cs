using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.U2D;

public class LoadFromAssetBundle : MonoBehaviour
{
    List<string> tags = new List<string>();

    void Awake()
    {
#if !ENABLE_WEB_REQUEST_ASSET_BUNDLE
        Debug.Log("Please enable UnityWebRequestAssetBundle Module from PackageManager");
#endif
    }

#if ENABLE_WEB_REQUEST_ASSET_BUNDLE
    void OnEnable()
    {
        SpriteAtlasManager.atlasRequested += RequestLateBindingAtlas;
        SpriteAtlasManager.atlasRegistered += AtlasRegistered;
    }

    void OnDisable()
    {
        SpriteAtlasManager.atlasRequested -= RequestLateBindingAtlas;
        SpriteAtlasManager.atlasRegistered -= AtlasRegistered;
    }

    void RequestLateBindingAtlas(string tag, System.Action<SpriteAtlas> action)
    {
        if (null == tags.FirstOrDefault(stringToCheck => stringToCheck.Contains(tag)))
        {
            tags.Add(tag);
            StartCoroutine(LoadAssetBundle(tag, action));
        }
    }

    IEnumerator LoadAssetBundle(string tag, System.Action<SpriteAtlas> callback)
    {
        var assetbundleToLoad = "atlasbundle";

        UnityWebRequest loadOp;
        if (Application.platform == RuntimePlatform.Android)
        {
            loadOp = UnityWebRequestAssetBundle.GetAssetBundle(Application.streamingAssetsPath + "/" + assetbundleToLoad);
        }
        else
        {
            loadOp = UnityWebRequestAssetBundle.GetAssetBundle("file://" + Application.streamingAssetsPath + "/" + assetbundleToLoad);
        }
        
        yield return loadOp.SendWebRequest();

        if (loadOp.result != UnityWebRequest.Result.Success)
        {
            Debug.Log(loadOp.error);
        }
        else
        {
            var ab = DownloadHandlerAssetBundle.GetContent(loadOp);
            if (null != ab)
            {
                var sa = ab.LoadAsset<SpriteAtlas>("fromassetbundle.spriteatlasv2");
                callback(sa);
                Debug.Log("AssetBundle : " + tag + " has Atlas " + sa.name);
            }
        }
    }

    void AtlasRegistered(SpriteAtlas spriteAtlas)
    {
        Debug.LogFormat("Registered {0}.", spriteAtlas.name);
    }
#endif
}
