using System;
using System.ComponentModel;
using Unity.Plastic.Newtonsoft.Json;

// Internal usage. This isn't a public API.
[EditorBrowsable(EditorBrowsableState.Never)]
[Obsolete("OrganizationCredentials is deprecated and will be removed in a future release", false)]
public class OrganizationCredentials
{
    // Internal usage. This isn't a public API.
    [EditorBrowsable(EditorBrowsableState.Never)]
    [JsonProperty("user")]
    public string User { get; set; }

    // Internal usage. This isn't a public API.
    [EditorBrowsable(EditorBrowsableState.Never)]
    [JsonProperty("password")]
    public string Password { get; set; }
}

