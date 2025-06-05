using UnityEngine;
public static class JsonUtilityWrapper
{
    public static T FromJson<T>(string json)
    {
        return JsonUtility.FromJson<Wrapper<T>>(WrapJson(json)).Data;
    }

    private static string WrapJson(string json) => "{\"Data\":" + json + "}";

    [System.Serializable]
    private class Wrapper<T>
    {
        public T Data;
    }
}
