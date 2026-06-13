import pytest
from llm_go.data.go_best_practices import GoProjectTemplates, GoLayoutValidator


class TestGoLayoutValidator:
    def test_correct_layout(self):
        paths = [
            "cmd/server/main.go",
            "cmd/worker/main.go",
            "internal/config/config.go",
            "internal/handler/handler.go",
            "go.mod",
        ]
        result = GoLayoutValidator.validate(paths)
        assert result["has_cmd_at_root"]  is True
        assert result["cmd_has_main_go"]  is True
        assert result["no_nested_cmd"]    is True
        assert GoLayoutValidator.layout_score(result) == 1.0

    def test_missing_cmd(self):
        paths = ["main.go", "internal/config/config.go", "go.mod"]
        result = GoLayoutValidator.validate(paths)
        assert result["has_cmd_at_root"]  is False
        assert result["cmd_has_main_go"]  is False

    def test_nested_cmd_flagged(self):
        paths = [
            "cmd/server/main.go",
            "internal/tools/cmd/helper/main.go",  # nested — bad practice
        ]
        result = GoLayoutValidator.validate(paths)
        assert result["no_nested_cmd"] is False

    def test_layout_score_range(self):
        paths = ["cmd/server/main.go"]
        score = GoLayoutValidator.layout_score(GoLayoutValidator.validate(paths))
        assert 0.0 <= score <= 1.0


class TestGoProjectTemplates:
    def test_generates_examples(self):
        tpl      = GoProjectTemplates()
        examples = tpl.all_examples()
        assert len(examples) > 50

    def test_cmd_main_contains_package_main(self):
        tpl = GoProjectTemplates()
        ex  = tpl._cmd_main("1.22", "server", "Server")
        assert "package main" in ex

    def test_cmd_main_respects_layout_rule(self):
        tpl = GoProjectTemplates()
        ex  = tpl._cmd_main("1.22", "server", "Server")
        # Must mention cmd/server or cmd/<app> path
        assert "cmd/" in ex or "cmd/" in ex

    def test_fiber_server_template(self):
        tpl = GoProjectTemplates()
        ex  = tpl._fiber_server("1.24", "api")
        assert "fiber.New" in ex
        assert "package main" in ex
        assert "cmd/" in ex

    def test_cobra_main_template(self):
        tpl = GoProjectTemplates()
        ex  = tpl._cobra_main("1.24", "cli")
        assert "cmd.Execute" in ex
        assert "package main" in ex

    def test_go_project_layout_constant(self):
        from llm_go.data.go_best_practices import GO_PROJECT_LAYOUT
        assert "cmd/" in GO_PROJECT_LAYOUT
        assert "ALWAYS at root" in GO_PROJECT_LAYOUT
        assert "internal/" in GO_PROJECT_LAYOUT
