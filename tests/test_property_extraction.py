import unittest

from imperia_matching_tfm.property_extraction.rule_based import extract_property_from_text


class PropertyExtractionTest(unittest.TestCase):
    def test_extracts_structured_fields_from_listing_text(self) -> None:
        text = """
        Piso en Calle de Fontanars dels Alforins, Vara de Quart, València
        PRECIO DE VENTA: 290.000€ con garaje opcional
        103 m² construidos, 95 m² útiles
        4 habitaciones
        2 baños
        Balcón
        Segunda mano/buen estado
        Construido en 1981
        No dispone de calefacción
        Planta 4ª exterior
        Con ascensor
        Muy luminoso. Buena distribución. Ideal para familias.
        """

        property_ = extract_property_from_text(
            text,
            source_id="idealista_111260654",
            source_url="https://www.idealista.com/pro/magnusrealestate/inmueble/111260654/",
        )

        self.assertEqual(property_["price"], 290000)
        self.assertEqual(property_["bedrooms"], 4)
        self.assertEqual(property_["bathrooms"], 2)
        self.assertEqual(property_["surface_m2"], 103)
        self.assertEqual(property_["usable_surface_m2"], 95)
        self.assertEqual(property_["built_year"], 1981)
        self.assertEqual(property_["floor"], 4)
        self.assertIn("elevator", property_["features"])
        self.assertIn("terrace", property_["features"])
        self.assertIn("muy_luminoso", property_["features"])

    def test_extracts_singular_bathroom_and_negative_elevator(self) -> None:
        text = """
        Piso en venta en Calle del Mestre Serrano
        Zona Centro, Xirivella
        183.000 €
        65 m² 3 hab. Planta 2ª exterior sin ascensor
        3 habitaciones
        1 baño completo
        Reformado
        Cocina independiente
        Segunda mano/buen estado
        Construido en 1964
        No dispone de calefacción
        Sin ascensor
        """

        property_ = extract_property_from_text(text, source_id="pdf2")

        self.assertEqual(property_["price"], 183000)
        self.assertEqual(property_["city"], "Xirivella")
        self.assertEqual(property_["zone"], "Zona Centro")
        self.assertEqual(property_["bathrooms"], 1)
        self.assertEqual(property_["built_year"], 1964)
        self.assertIn("sin_ascensor", property_["features"])
        self.assertIn("reformado", property_["features"])
        self.assertNotIn("elevator", property_["features"])

    def test_extracts_alicante_location_and_does_not_confuse_estrenar_with_atico(self) -> None:
        text = """
        Piso en venta en Calle del Catedrático Daniel Jiménez de Cisneros, 7
        Carolinas Altas, Alicante / Alacant
        155.000 €
        67 m² 2 hab. Bajo exterior sin ascensor
        Bajo reformado a estrenar con entrada privada desde la calle en el barrio de Carolinas.
        2 habitaciones
        1 baño
        Terraza
        67 m² construidos, 60 m² útiles
        Construido en 1968
        Barrio Carolinas Altas
        Distrito Campoamor-Carolinas-Altozano
        Alicante / Alacant
        """

        property_ = extract_property_from_text(text, source_id="pdf4")

        self.assertEqual(property_["property_type"], "FLAT")
        self.assertEqual(property_["city"], "Alicante / Alacant")
        self.assertEqual(property_["zone"], "Carolinas Altas")
        self.assertEqual(property_["province"], "Alicante")
        self.assertEqual(property_["bathrooms"], 1)
        self.assertEqual(property_["surface_m2"], 67)
        self.assertEqual(property_["usable_surface_m2"], 60)
        self.assertIn("sin_ascensor", property_["features"])


if __name__ == "__main__":
    unittest.main()
