        telegraph_url = _festival_telegraph_url(fest)
        lines = ["Иллюстрации фестиваля"]
        if telegraph_url:
            lines.append(telegraph_url)
        lines.extend(
            [
                f"Всего: {total}",
                f"Текущая обложка: #{current}",
                "Выберите новое изображение обложки:",
            ]
        text = "\n".join(lines)
