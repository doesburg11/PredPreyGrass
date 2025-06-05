using UnityEngine;

public class TooltipTrigger : MonoBehaviour
{
    public string tooltipText = "Default Tooltip";

    void OnMouseEnter()
    {
        Debug.Log("Hovering over " + gameObject.name);
        TooltipManager.Instance.ShowTooltip(tooltipText);
    }

    void OnMouseExit()
    {
        TooltipManager.Instance.HideTooltip();
    }
}
