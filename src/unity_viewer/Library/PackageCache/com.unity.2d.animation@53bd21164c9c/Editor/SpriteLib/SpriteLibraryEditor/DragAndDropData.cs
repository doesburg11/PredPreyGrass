using System;
using System.Collections.Generic;
using UnityEngine;

namespace UnityEditor.U2D.Animation.SpriteLibraryEditor
{
    internal enum SpriteSourceType
    {
        Sprite,
        Psb
    }

    internal struct DragAndDropData
    {
        public SpriteSourceType spriteSourceType;
        public string name;
        public List<Sprite> sprites;
    }
}
