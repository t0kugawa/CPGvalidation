from t2xcpg.bs import BaseDrive, BrainStem


def test_bs_and_base_relays_are_linked(nest_kernel):
    from persona.registry import Registry

    registry = Registry()
    BrainStem(registry=registry, n_bs=4)
    BaseDrive(registry=registry, n_bs=4)
    links = registry.relay_links()
    assert links["BS_in_e"] == ("bs_pg_e", "BS_E")
    assert links["BS_in_f"] == ("bs_pg_f", "BS_F")
    assert links["BASE_in"] == ("base_pg", "BASE_pg")
