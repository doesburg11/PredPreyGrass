#if UNITY_EDITOR
using System;

namespace UnityEngine.U2D.Animation
{
    public partial class SpriteResolver : ISerializationCallbackReceiver
    {
        internal static string spriteHashPropertyName => nameof(m_SpriteHash);

        bool m_SpriteLibChanged;

        /// <summary>
        /// Raised when object is deserialized in the Editor.
        /// </summary>
        public event Action onDeserializedCallback = () => { };

        long m_SpriteLibraryModificationHash;
        internal long spriteLibraryModificationHash => m_SpriteLibraryModificationHash;

        void LateUpdateEditor()
        {
            var newSpriteLibraryModificationHash = GetCurrentSpriteLibraryAssetModificationHash();
            if (m_SpriteLibraryModificationHash != newSpriteLibraryModificationHash)
            {
                ResolveSpriteToSpriteRenderer();
                spriteLibChanged = true;
                m_SpriteLibraryModificationHash = newSpriteLibraryModificationHash;
            }
        }

        void OnDidApplyAnimationProperties()
        {
            if (IsInGUIUpdateLoop())
                ResolveUpdatedValue();
        }

        internal bool spriteLibChanged
        {
            get => m_SpriteLibChanged;
            set => m_SpriteLibChanged = value;
        }

        /// <summary>
        /// Called before object is serialized.
        /// </summary>
        void ISerializationCallbackReceiver.OnBeforeSerialize() { }

        /// <summary>
        /// Called after object is deserialized.
        /// </summary>
        void ISerializationCallbackReceiver.OnAfterDeserialize()
        {
            onDeserializedCallback();
        }

        long GetCurrentSpriteLibraryAssetModificationHash()
        {
            if (spriteLibrary != null)
            {
                var spriteLibraryAsset = spriteLibrary.spriteLibraryAsset;
                if (spriteLibraryAsset != null)
                    return spriteLibraryAsset.modificationHash;
            }

            return 0;
        }
    }
}
#endif
