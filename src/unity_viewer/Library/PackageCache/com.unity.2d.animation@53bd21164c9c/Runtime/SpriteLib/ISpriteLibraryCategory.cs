using System.Collections.Generic;

namespace UnityEngine.U2D.Animation
{
    /// <summary>
    /// Represents a Sprite Library's category.
    /// </summary>
    public interface ISpriteLibraryCategory
    {
        /// <summary>
        /// Category's name.
        /// </summary>
        string name { get; }

        /// <summary>
        /// Labels contained in the Category.
        /// </summary>
        IEnumerable<ISpriteLibraryLabel> labels { get; }
    }
}
