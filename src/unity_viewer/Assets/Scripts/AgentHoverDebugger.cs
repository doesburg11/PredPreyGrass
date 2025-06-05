using UnityEngine;

public class AgentHoverDebugger : MonoBehaviour
{
    public float energy = 0f; // This will be filled by JsonGridStepPlayer

    void OnMouseEnter()
    {
        Vector3 pos = transform.position;
        string message = $"Energy: {energy:F2}\nPos: ({(int)pos.x}, {(int)pos.y})";
        if (TooltipManager.Instance != null)
            TooltipManager.Instance.ShowTooltip(message);
    }

    void OnMouseExit()
    {
        if (TooltipManager.Instance != null)
            TooltipManager.Instance.HideTooltip();
    }
}
