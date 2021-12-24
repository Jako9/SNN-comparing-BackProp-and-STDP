def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        batch_size = self.source.batch_size

        source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
        source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
        target_s = self.target.s.view(batch_size, -1).unsqueeze(1).float()
        target_x = self.target.x.view(batch_size, -1).unsqueeze(1)

        update = 0

        # Pre-synaptic update.
        if self.nu[0]:
            outer_product = self.reduction(torch.bmm(source_s, target_x), dim=0)
            update -= self.nu[0] * outer_product * (self.connection.w - self.wmin)

        # Post-synaptic update.
        if self.nu[1]:
            outer_product = self.reduction(torch.bmm(source_x, target_s), dim=0)
            update += self.nu[1] * outer_product * (self.wmax - self.connection.w)

        self.connection.w += update

        super().update()
