from predpreygrass.malthusian_rl.test.strict_malthusian_smoke import run_smoke_validation


def test_strict_malthusian_smoke():
    result = run_smoke_validation()

    assert result["strict_mode"] == "strict"
    assert result["mu_update"] == "multiplicative"
    assert result["within_episode_reproduction"] is False
    assert result["islands"] == 4
    assert set(result["mu_before"].keys()) == {
        "type_1_predator",
        "type_2_predator",
        "type_1_prey",
        "type_2_prey",
    }
    assert set(result["mu_after"].keys()) == set(result["mu_before"].keys())
    for species, island_map in result["mu_before"].items():
        assert len(island_map) == 4
        assert len(result["mu_after"][species]) == 4
    assert set(result["phi_by_species"].keys()) == set(result["mu_before"].keys())