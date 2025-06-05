using UnityEngine;

namespace UnityEditor.U2D.Common.Path
{
    internal interface IUndoObject
    {
        void RegisterUndo(string name);
    }
}
