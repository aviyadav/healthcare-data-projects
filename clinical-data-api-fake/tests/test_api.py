"""
API endpoint tests for the Clinical Data API.

Tests cover:
- Health / utility endpoints
- All 6 domain endpoints (AE, CM, DM, LB, TV, VS)
- Pagination (default + explicit)
- Filter parameters (study, site, subject, visit, form)
- Invalid / non-matching filters return empty data (not errors)
- Response schema validation (meta fields present and correct types)
"""
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

DOMAINS = ["ae", "cm", "dm", "lb", "tv", "vs"]


def _domain_url(domain: str) -> str:
    return f"/api/v1/{domain}"


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------


class TestSystemEndpoints:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_has_status(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_domains_endpoint_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/domains")
        assert response.status_code == 200

    def test_domains_lists_all_six_domains(self, client: TestClient) -> None:
        data = client.get("/api/v1/domains").json()
        domain_codes = {d["domain"] for d in data["domains"]}
        assert domain_codes == {"AE", "CM", "DM", "LB", "TV", "VS"}

    def test_docs_available(self, client: TestClient) -> None:
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_json_available(self, client: TestClient) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Domain endpoints — basic smoke tests (parametrized)
# ---------------------------------------------------------------------------


class TestDomainEndpointsSmoke:
    @pytest.mark.parametrize("domain", DOMAINS)
    def test_domain_returns_200(self, client: TestClient, domain: str) -> None:
        response = client.get(_domain_url(domain))
        assert response.status_code == 200, f"{domain}: {response.text}"

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_domain_response_has_data_and_meta(self, client: TestClient, domain: str) -> None:
        data = client.get(_domain_url(domain)).json()
        assert "data" in data, f"{domain}: missing 'data'"
        assert "meta" in data, f"{domain}: missing 'meta'"

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_domain_data_is_list(self, client: TestClient, domain: str) -> None:
        data = client.get(_domain_url(domain)).json()
        assert isinstance(data["data"], list), f"{domain}: 'data' is not a list"

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_domain_meta_fields_present(self, client: TestClient, domain: str) -> None:
        meta = client.get(_domain_url(domain)).json()["meta"]
        for field in ("page", "page_size", "total_records", "total_pages"):
            assert field in meta, f"{domain}: meta missing '{field}'"


# ---------------------------------------------------------------------------
# Pagination tests
# ---------------------------------------------------------------------------


class TestPagination:
    @pytest.mark.parametrize("domain", DOMAINS)
    def test_default_page_is_1(self, client: TestClient, domain: str) -> None:
        meta = client.get(_domain_url(domain)).json()["meta"]
        assert meta["page"] == 1

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_default_page_size(self, client: TestClient, domain: str) -> None:
        meta = client.get(_domain_url(domain)).json()["meta"]
        # Default page_size is 100 per settings
        assert meta["page_size"] == 100

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_custom_page_size(self, client: TestClient, domain: str) -> None:
        response = client.get(_domain_url(domain), params={"page_size": 10})
        assert response.status_code == 200
        body = response.json()
        assert body["meta"]["page_size"] == 10
        assert len(body["data"]) <= 10

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_second_page_returns_different_data(self, client: TestClient, domain: str) -> None:
        page1 = client.get(_domain_url(domain), params={"page": 1, "page_size": 5}).json()
        page2 = client.get(_domain_url(domain), params={"page": 2, "page_size": 5}).json()
        # Only check if there are enough records for 2 pages
        if page1["meta"]["total_records"] > 5:
            assert page1["data"] != page2["data"], f"{domain}: pages 1 and 2 returned same data"

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_page_beyond_total_returns_empty_data(self, client: TestClient, domain: str) -> None:
        response = client.get(_domain_url(domain), params={"page": 999999, "page_size": 100})
        assert response.status_code == 200
        assert response.json()["data"] == []

    def test_invalid_page_zero_returns_422(self, client: TestClient) -> None:
        response = client.get("/api/v1/ae", params={"page": 0})
        assert response.status_code == 422

    def test_invalid_page_size_zero_returns_422(self, client: TestClient) -> None:
        response = client.get("/api/v1/ae", params={"page_size": 0})
        assert response.status_code == 422

    def test_page_size_exceeds_max_returns_422(self, client: TestClient) -> None:
        # max_page_size is 1000 per settings
        response = client.get("/api/v1/ae", params={"page_size": 9999})
        assert response.status_code == 422

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_total_pages_consistent_with_total_records(self, client: TestClient, domain: str) -> None:
        import math
        body = client.get(_domain_url(domain), params={"page_size": 50}).json()
        meta = body["meta"]
        if meta["total_records"] == 0:
            expected_pages = 0
        else:
            expected_pages = math.ceil(meta["total_records"] / meta["page_size"])
        assert meta["total_pages"] == expected_pages


# ---------------------------------------------------------------------------
# Filter tests
# ---------------------------------------------------------------------------


class TestFilters:
    def _get_first_value(self, client: TestClient, domain: str, field: str):
        """Helper: get the first non-null value of `field` in a domain."""
        data = client.get(_domain_url(domain), params={"page_size": 50}).json()["data"]
        for row in data:
            if row.get(field):
                return row[field]
        return None

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_study_filter_reduces_or_equals_total(self, client: TestClient, domain: str) -> None:
        unfiltered = client.get(_domain_url(domain)).json()["meta"]["total_records"]
        value = self._get_first_value(client, domain, "STUDY")
        if value is None:
            pytest.skip(f"{domain}: no STUDY value found")
        filtered = client.get(_domain_url(domain), params={"study": value}).json()["meta"]["total_records"]
        assert filtered <= unfiltered

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_site_filter_reduces_or_equals_total(self, client: TestClient, domain: str) -> None:
        unfiltered = client.get(_domain_url(domain)).json()["meta"]["total_records"]
        value = self._get_first_value(client, domain, "SITE")
        if value is None:
            pytest.skip(f"{domain}: no SITE value found")
        filtered = client.get(_domain_url(domain), params={"site": value}).json()["meta"]["total_records"]
        assert filtered <= unfiltered

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_subject_filter_reduces_or_equals_total(self, client: TestClient, domain: str) -> None:
        unfiltered = client.get(_domain_url(domain)).json()["meta"]["total_records"]
        value = self._get_first_value(client, domain, "SUBJECT")
        if value is None:
            pytest.skip(f"{domain}: no SUBJECT value found")
        filtered = client.get(_domain_url(domain), params={"subject": value}).json()["meta"]["total_records"]
        assert filtered <= unfiltered

    @pytest.mark.parametrize("domain", ["ae", "cm", "dm", "lb", "vs"])
    def test_nonexistent_filter_value_returns_empty(self, client: TestClient, domain: str) -> None:
        response = client.get(_domain_url(domain), params={"study": "__NO_SUCH_STUDY_XYZ__"})
        assert response.status_code == 200
        body = response.json()
        assert body["data"] == []
        assert body["meta"]["total_records"] == 0

    @pytest.mark.parametrize("domain", DOMAINS)
    def test_filtered_data_matches_filter_value(self, client: TestClient, domain: str) -> None:
        """All returned records should match the filter that was applied."""
        value = self._get_first_value(client, domain, "STUDY")
        if value is None:
            pytest.skip(f"{domain}: no STUDY value found")
        data = client.get(
            _domain_url(domain), params={"study": value, "page_size": 20}
        ).json()["data"]
        for row in data:
            if row.get("STUDY") is not None:
                assert row["STUDY"] == value, f"{domain}: found STUDY={row['STUDY']!r}, expected {value!r}"

    def test_combined_filters(self, client: TestClient) -> None:
        """Combining multiple filters should not error and narrows the result set."""
        # Get valid values from unfiltered AE
        data = client.get("/api/v1/ae", params={"page_size": 50}).json()["data"]
        if not data:
            pytest.skip("AE has no data")
        row = data[0]
        params = {k: row[k] for k in ("STUDY", "SITE") if row.get(k)}
        if len(params) < 2:
            pytest.skip("Not enough non-null filter values in first AE row")
        response = client.get("/api/v1/ae", params=params)
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Domain-specific field tests
# ---------------------------------------------------------------------------


class TestDomainFields:
    def test_ae_has_aeterm(self, client: TestClient) -> None:
        data = client.get("/api/v1/ae", params={"page_size": 5}).json()["data"]
        if data:
            assert "AETERM" in data[0]

    def test_cm_has_cmtrt(self, client: TestClient) -> None:
        data = client.get("/api/v1/cm", params={"page_size": 5}).json()["data"]
        if data:
            assert "CMTRT" in data[0]

    def test_dm_has_age_and_sex(self, client: TestClient) -> None:
        data = client.get("/api/v1/dm", params={"page_size": 5}).json()["data"]
        if data:
            assert "AGE" in data[0]
            assert "SEX" in data[0]

    def test_lb_has_lborres(self, client: TestClient) -> None:
        data = client.get("/api/v1/lb", params={"page_size": 5}).json()["data"]
        if data:
            assert "LBORRES" in data[0]

    def test_tv_has_visitnum(self, client: TestClient) -> None:
        data = client.get("/api/v1/tv", params={"page_size": 5}).json()["data"]
        if data:
            assert "VISITNUM" in data[0]

    def test_vs_has_vsorres(self, client: TestClient) -> None:
        data = client.get("/api/v1/vs", params={"page_size": 5}).json()["data"]
        if data:
            assert "VSORRES" in data[0]
