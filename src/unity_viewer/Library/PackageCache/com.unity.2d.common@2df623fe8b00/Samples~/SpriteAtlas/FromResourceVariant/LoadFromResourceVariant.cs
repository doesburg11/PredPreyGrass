using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.U2D;

public class LoadFromResourceVariant : MonoBehaviour
{
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

    void RequestLateBindingAtlas(string tag, System.Action<SpriteAtlas> callback)
    {
        if (tag == "ResourceAtlas2")
        {
            var sa = UnityEngine.Resources.Load<SpriteAtlas>("ResourceAtlasVariant2");
            callback(sa);
        }
        else
            Debug.Log("Error: Late binding callback with wrong atlas tag of " + tag);
    }

    void AtlasRegistered(SpriteAtlas spriteAtlas)
    {
        Debug.LogFormat("Registered {0}.", spriteAtlas.name);
    }
}
