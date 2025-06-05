using System.Collections.Generic;
using UnityEditor.U2D.Sprites;

namespace UnityEditor.U2D.Common
{
    internal class SpriteCustomDataProvider
#if SPRITE_CUSTOM_DATA
        : ISpriteCustomDataProvider
#endif
    {
        public static bool HasDataProvider(ISpriteEditorDataProvider dataProvider)
        {
#if SPRITE_CUSTOM_DATA
            return dataProvider.HasDataProvider(typeof(ISpriteCustomDataProvider));
#else
            return false;
#endif
        }

#if SPRITE_CUSTOM_DATA

        readonly ISpriteCustomDataProvider m_DataProvider;

        public SpriteCustomDataProvider(ISpriteEditorDataProvider dataProvider)
        {
            m_DataProvider = dataProvider.GetDataProvider<ISpriteCustomDataProvider>();
        }

        public IEnumerable<string> GetKeys() => m_DataProvider.GetKeys();

        public void SetData(string key, string data) => m_DataProvider.SetData(key, data);

        public void RemoveData(string key) => m_DataProvider.RemoveData(key);

        public bool GetData(string key, out string data) => m_DataProvider.GetData(key, out data);

#endif
    }
}
