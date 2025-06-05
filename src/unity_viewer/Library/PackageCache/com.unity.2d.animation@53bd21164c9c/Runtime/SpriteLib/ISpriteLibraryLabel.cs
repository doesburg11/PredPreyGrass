namespace UnityEngine.U2D.Animation
{
    /// <summary>
    /// Represents a Sprite Library's label.
    /// </summary>
    public interface ISpriteLibraryLabel
    {
        /// <summary>
        /// Label's name.
        /// </summary>
        string name { get; }

        /// <summary>
        /// Label's Sprite.
        /// </summary>
        Sprite sprite { get; }
    }
}
