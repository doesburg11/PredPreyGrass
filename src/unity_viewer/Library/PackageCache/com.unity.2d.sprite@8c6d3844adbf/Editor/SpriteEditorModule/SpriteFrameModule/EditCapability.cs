using System;
using UnityEngine;

namespace UnityEditor.U2D.Sprites
{
    /// <summary>
    /// Structure to hold the edit capability of a sprite in Sprite Editor Window.
    /// </summary>
    [Serializable]
    public struct EditCapability
    {
        /// <summary>
        /// Default editing capability where all capability is allowed.
        /// </summary>
        public static EditCapability defaultCapability = new EditCapability() { m_Capability = EEditCapability.None };

        [SerializeField]
        EEditCapability m_Capability;

        /// <summary>
        /// Constructor to create EditCapability with specific capabilities.
        /// </summary>
        /// <param name="capabilities">Capabilities allowed.</param>
        public EditCapability(params EEditCapability[] capabilities)
        {
            m_Capability = EEditCapability.None;
            if (capabilities != null)
            {
                for (int i = 0; i < capabilities.Length; ++i)
                {
                    SetCapability(capabilities[i], true);
                }
            }
        }

        /// <summary>
        /// Check if the capability is enabled.
        /// </summary>
        /// <param name="hasCapability">Capability to check.</param>
        /// <returns>True if capability is enabled. False otherwise.</returns>
        public bool HasCapability(EEditCapability hasCapability)
        {
            return m_Capability.HasFlag(hasCapability);
        }

        /// <summary>
        /// Enable or disable a capability.
        /// </summary>
        /// <param name="capability">Capability to modify.</param>
        /// <param name="on">True to enable. False otherwise.</param>
        public void SetCapability(EEditCapability capability, bool on)
        {
            if(on)
                m_Capability |= capability;
            else
                m_Capability &= ~capability;
        }
    }

    /// <summary>
    /// Edit capability flags.
    /// </summary>
    [Flags]
    public enum EEditCapability
    {
        /// <summary> No capability. </summary>
        None = 0,
        /// <summary> Edit sprite's name capability. </summary>
        EditSpriteName = 1 << 0,
        /// <summary> Edit sprite's rect capability. </summary>
        EditSpriteRect = 1 << 1,
        /// <summary> Edit sprite's border capability. </summary>
        EditBorder = 1 << 2,
        /// <summary> Edit sprite's pivot capability. </summary>
        EditPivot = 1 << 3,
        /// <summary> Allow creation and deletion of sprite.</summary>
        CreateAndDeleteSprite = 1 << 4,
        /// <summary>Allow slicing of Sprites on import of asset.</summary>
        SliceOnImport = 1 << 5,
        /// <summary> All capabilities are enabled.</summary>
        All = EditSpriteRect | EditBorder | EditPivot | EditSpriteName | CreateAndDeleteSprite | SliceOnImport
    }
}
