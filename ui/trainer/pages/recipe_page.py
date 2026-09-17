"""Recipe creation and management page of the trainer."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from services.recipe_service import RecipeService
from ui.theme import (
    SPACE_L,
    SPACE_M,
    SPACE_S,
    card_frame,
    make_button,
    section_label,
    title_label,
)

PACKAGE_FAMILIES = ["SO", "QFN", "QFP", "BGA", "DIP"]


class RecipePage(QWidget):
    """Lists recipes and edits their package metadata."""

    def __init__(self):
        super().__init__()

        self.recipe_service = RecipeService()
        self.selected_recipe = None
        self.parent_window = None

        self.setup_ui()
        self.load_recipe_list()

    def setup_ui(self):
        root_layout = QHBoxLayout()
        root_layout.setContentsMargins(SPACE_L, SPACE_L, SPACE_L, SPACE_L)
        root_layout.setSpacing(SPACE_L)

        # ----- Left panel: recipe list -----
        list_card = card_frame()
        list_layout = QVBoxLayout(list_card)
        list_layout.setContentsMargins(SPACE_L, SPACE_L, SPACE_L, SPACE_L)
        list_layout.setSpacing(SPACE_M)

        list_layout.addWidget(section_label("Recipes"))

        self.recipe_list = QListWidget()
        self.recipe_list.itemClicked.connect(self.load_selected_recipe)
        list_layout.addWidget(self.recipe_list, 1)

        self.btn_add_recipe = make_button("+  Add Recipe")
        self.btn_add_recipe.clicked.connect(self.new_recipe)
        list_layout.addWidget(self.btn_add_recipe)

        self.btn_delete_recipe = make_button("Delete Recipe", "danger")
        self.btn_delete_recipe.clicked.connect(self.delete_recipe)
        list_layout.addWidget(self.btn_delete_recipe)

        # ----- Right panel: recipe form -----
        form_card = card_frame()
        form_layout = QVBoxLayout(form_card)
        form_layout.setContentsMargins(SPACE_L, SPACE_L, SPACE_L, SPACE_L)
        form_layout.setSpacing(SPACE_M)

        form_layout.addWidget(title_label("Recipe Configuration"))

        self.recipe_name = QLineEdit()
        self.package_family = QComboBox()
        self.package_family.addItems(PACKAGE_FAMILIES)
        self.package_type = QLineEdit()
        self.package_size = QLineEdit()
        self.package_version = QLineEdit()
        self.type_name = QLineEdit()
        self.pin_count = QLineEdit()
        self.notes = QTextEdit()

        fields = QFormLayout()
        fields.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        fields.setHorizontalSpacing(SPACE_M)
        fields.setVerticalSpacing(SPACE_S)
        fields.addRow("Recipe Name", self.recipe_name)
        fields.addRow("Package Family", self.package_family)
        fields.addRow("Package Type", self.package_type)
        fields.addRow("Package Size", self.package_size)
        fields.addRow("Package Version", self.package_version)
        fields.addRow("Type Name", self.type_name)
        fields.addRow("Pin Count", self.pin_count)
        form_layout.addLayout(fields)

        notes_label = section_label("Notes")
        form_layout.addWidget(notes_label)
        form_layout.addWidget(self.notes, 1)

        self.btn_save_recipe = make_button("Save Recipe")
        self.btn_save_recipe.clicked.connect(self.save_recipe)
        form_layout.addWidget(self.btn_save_recipe)

        root_layout.addWidget(list_card, 1)
        root_layout.addWidget(form_card, 3)

        self.setLayout(root_layout)

    # ------------------------------------------------------------------
    # Recipe list
    # ------------------------------------------------------------------

    def load_recipe_list(self):
        self.recipe_list.clear()
        self.recipe_list.addItems(self.recipe_service.get_recipe_list())

    def load_selected_recipe(self, item):
        recipe_name = item.text()
        recipe = self.recipe_service.load_recipe(recipe_name)
        if not recipe:
            return

        self.selected_recipe = recipe_name

        self.recipe_name.setText(recipe.get("recipe_name", ""))
        self.package_type.setText(recipe.get("package_type", ""))
        self.package_size.setText(recipe.get("package_size", ""))
        self.package_version.setText(recipe.get("package_version", ""))
        self.type_name.setText(recipe.get("type_name", ""))
        self.pin_count.setText(str(recipe.get("pin_count", 0)))
        self.notes.setPlainText(recipe.get("notes", ""))

        family = recipe.get("package_family", "SO")
        index = self.package_family.findText(family)
        if index >= 0:
            self.package_family.setCurrentIndex(index)

        if self.parent_window:
            self.parent_window.update_context(recipe)

        print(f"[RECIPE] Selected {recipe_name}")

    # ------------------------------------------------------------------
    # Create, save and delete
    # ------------------------------------------------------------------

    def new_recipe(self):
        """Clear the form for a new recipe."""
        self.selected_recipe = None

        self.recipe_name.clear()
        self.package_type.clear()
        self.package_size.clear()
        self.package_version.clear()
        self.type_name.clear()
        self.pin_count.clear()
        self.notes.clear()

    def save_recipe(self):
        recipe_name = self.recipe_name.text().strip()
        if not recipe_name:
            QMessageBox.warning(self, "Warning", "Recipe Name is required.")
            return

        try:
            pin_count = int(self.pin_count.text() or 0)
        except ValueError:
            QMessageBox.warning(self, "Warning", "Pin Count must be a whole number.")
            return

        try:
            if self.recipe_service.recipe_exists(recipe_name):
                try:
                    recipe = self.recipe_service.load_recipe(recipe_name)
                except ValueError as error:
                    QMessageBox.critical(
                        self, "Error", f"Existing recipe file is corrupted: {error}"
                    )
                    return
                if recipe is None:
                    recipe = self.recipe_service.get_recipe_template()
            else:
                recipe = self.recipe_service.get_recipe_template()

            recipe["recipe_name"] = recipe_name
            recipe["package_family"] = self.package_family.currentText()
            recipe["package_type"] = self.package_type.text()
            recipe["package_size"] = self.package_size.text()
            recipe["package_version"] = self.package_version.text()
            recipe["type_name"] = self.type_name.text()
            recipe["pin_count"] = pin_count
            recipe["notes"] = self.notes.toPlainText()

            if self.recipe_service.recipe_exists(recipe_name):
                self.recipe_service.save_recipe(recipe_name, recipe)
            else:
                self.recipe_service.create_recipe(recipe_name, recipe)
        except ValueError as error:
            QMessageBox.warning(
                self, "Warning", f"Recipe name is not valid: {error}"
            )
            return

        self.load_recipe_list()
        self.selected_recipe = recipe_name

        if self.parent_window:
            self.parent_window.update_context(recipe)
            if hasattr(self.parent_window, "refresh_step_bar"):
                self.parent_window.refresh_step_bar()

        QMessageBox.information(self, "Success", "Recipe saved successfully.")

    def delete_recipe(self):
        if not self.selected_recipe:
            QMessageBox.warning(self, "Warning", "Please select a recipe.")
            return

        try:
            self.recipe_service.delete_recipe(self.selected_recipe)
        except ValueError as error:
            QMessageBox.warning(
                self, "Warning", f"Recipe name is not valid: {error}"
            )
            return

        self.selected_recipe = None
        self.new_recipe()
        self.load_recipe_list()

        if self.parent_window and hasattr(self.parent_window, "clear_context"):
            self.parent_window.clear_context()

        QMessageBox.information(self, "Success", "Recipe deleted.")
